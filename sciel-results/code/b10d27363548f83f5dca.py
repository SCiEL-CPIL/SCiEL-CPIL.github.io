# EVOLVE-BLOCK-START
import numpy as np

def priority(adjacency):
    """
    Computes a priority score for each vertex using spectral graph theory.
    
    Approach:
    1. Treat the adjacency matrix as a symmetric matrix.
    2. Compute the eigenvector corresponding to the largest eigenvalue (Perron-Frobenius).
    3. This eigenvector represents 'Eigenvector Centrality'.
    4. Vertices with high centrality are 'hubs' that block many neighbors.
    5. Vertices with low centrality are 'peripheral' and safer to pick.
    6. Score is inversely proportional to centrality.
    """
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=float)
        
    N = adjacency.shape[0]
    
    # Ensure symmetry for real eigenvalues
    A = (adjacency + adjacency.T) / 2.0
    
    # Compute eigenvalues and eigenvectors
    # For N=200, this is very fast
    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)
    except np.linalg.LinAlgError:
        # Fallback to degree-based if eig fails (should not happen for symmetric)
        deg = A.sum(axis=1)
        return 1.0 / (deg + 1.0)
        
    # Find the index of the largest eigenvalue
    idx_max = np.argmax(eigenvalues)
    v = eigenvectors[:, idx_max]
    
    # Ensure the vector is positive (Eigenvector is defined up to sign)
    if np.sum(v) < 0:
        v = -v
        
    # Normalize to avoid numerical issues with very small/large values
    # We want to invert the values.
    # Add a small epsilon to avoid division by zero
    eps = 1e-6
    scores = 1.0 / (v + eps)
    
    # Optional: Combine with degree to break ties or refine
    # Low degree is also good.
    deg = A.sum(axis=1)
    # Normalize degree to [0,1] range roughly
    max_deg = np.max(deg) if np.max(deg) > 0 else 1
    deg_norm = deg / max_deg
    
    # Blend spectral score and degree score
    # Spectral score is already "higher is better" (inverse centrality)
    # Degree score: lower degree is better.
    deg_score = 1.0 / (deg_norm + 1.0)
    
    final_scores = 0.5 * scores + 0.5 * deg_score
    
    return final_scores
# EVOLVE-BLOCK-END
