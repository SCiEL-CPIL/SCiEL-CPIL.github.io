import numpy as np

def priority(adjacency):
    """
    Computes a priority score for each vertex using the principal eigenvector
    of the adjacency matrix. Vertices with lower centrality (lower eigenvector
    values) are assigned higher priority, as they are less likely to block
    other vertices in the independent set.
    """
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    
    # Ensure float type for eigendecomposition
    A = adjacency.astype(float)
    
    # Compute eigenvalues and eigenvectors
    # eigh is for symmetric matrices
    eigvals, eigvecs = np.linalg.eigh(A)
    
    # The largest eigenvalue is the last one
    principal_eigvec = eigvecs[:, -1]
    
    # We want to prioritize vertices with LOW centrality.
    # So we negate the eigenvector.
    score = -principal_eigvec
    
    return score
