"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh


def priority(adjacency):
    """
    Assign a score to each vertex based on its eigenvector centrality.
    Vertices with lower eigenvector centrality are considered more "peripheral"
    and are prioritized for selection in the greedy independent set algorithm.
    This avoids picking central hubs that would block many other vertices.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Convert to sparse matrix for efficiency
    A_sparse = csc_matrix(adj)
    
    # Handle edge case: empty graph
    if A_sparse.nnz == 0:
        return np.zeros(n)
    
    try:
        # Get the largest eigenvalue and its corresponding eigenvector
        # k=1 gets the largest magnitude eigenvalue
        vals, vecs = eigsh(A_sparse, k=1, which='LM', return_eigenvectors=True)
        # eigsh returns eigenvectors as columns
        centrality = vecs[:, 0]
        
        # Ensure the eigenvector is positive (Perron-Frobenius property)
        if np.sum(centrality) < 0:
            centrality = -centrality
            
        # Normalize to ensure scores are in a reasonable range
        max_c = np.max(centrality)
        if max_c > 0:
            centrality_norm = centrality / max_c
        else:
            centrality_norm = centrality
            
        # Score: lower centrality is better (higher score for greedy)
        # We return negative centrality so that low centrality -> high score
        score = -centrality_norm
        
        return score
        
    except Exception:
        # Fallback to degree-based if eigsh fails
        d = adj.sum(axis=1)
        return -d
# EVOLVE-BLOCK-END
