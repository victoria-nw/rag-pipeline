import pandas as pd
from langchain_community.document_loaders import TextLoader
from sentence_transformers import SentenceTransformer
import chromadb
import meta


def load_data(file_path):
    df = pd.read_csv(file_path)
    print(df.head())
    return df


# text processor
def create_chunks(df):
    chunks = []
    for index, row in df.iterrows():
        chunk = {
            'doc_id': row['doc_id'],
            'title': row['title'],
            'content': row['content'],
            'category': row['category']
        }
        chunks.append(chunk)
    return chunks


# embeddings
def generate_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    text = [chunk['content'] for chunk in chunks]

    embeddings = model.encode(text, show_progress_bar=True)

    print(f"Generated {len(embeddings)} embeddings.")
    print (f"Embedding dimension: {embeddings.shape}")

    return embeddings, chunks


# vector database
def create_vector_database(embeddings, chunks):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="documentation")

    ids = [chunk['doc_id'] for chunk in chunks]
    documents = [chunk['content'] for chunk in chunks]
    metadatas = [{'title': chunk['title'], 'category': chunk['category']} for chunk in chunks]
    embeddings_list = embeddings.tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings_list
    )
    
    print(f"Successfully added {len(chunks)} documents to the vector database.")
    return collection


# retrieval
def retrieve_documents(query, collection, top_k=3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=top_k
    )

    return results


# mock llm generation
def mock_llm_generation(query, results):
    """Simulate llm response"""

    documents = results['documents'][0]
    metadatas = results['metadatas'][0]

    context = "\n\n".join([f"- {doc}" for doc in documents])

    prompt = f"""Based on the following context, answer the question.
Context:
{context}

Question: {query}
Answer: Based on the provided documentation,"""

    answer = f"Based on the provided documentation, here's what i found: \n\n"
    for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
        answer += f"{i}. {meta['title']}: {doc}\n\n"

    return answer, prompt 


def rag_pipeline(query, collection, top_k=3):
    results = retrieve_documents(query, collection, top_k)
    answer, prompt = mock_llm_generation(query, results)
    return answer, prompt


def main():

    file_path = 'data.csv'

    print("Loading data...")
    df=load_data()

    print("Creating chunks")
    chunks=create_chunks(df)

    print("Generating embeddings...")
    embeddings=generate_embeddings(chunks)

    print("Creating vector database...")
    collection=create_vector_database(embeddings, chunks)

    test_queries=[
        "How do i install the software?",
        "What are the steps for user authentication?",
        "How do i handle network issues?"
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        answer, prompt = rag_pipeline(query, collection)
        print(answer)


if __name__ == "__main__":
    main()