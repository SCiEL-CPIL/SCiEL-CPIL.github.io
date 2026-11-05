"""Program using spectral centrality for Maximum Independent Set priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign scores to vertices based on eigenvector centrality.
    Vertices with lower eigenvector centrality (more peripheral) get higher priority.
    We use power iteration to compute the principal eigenvector of the adjacency matrix.
    
    Improvement:
    1. Initialize with degree vector for faster and more stable convergence.
    2. Use an inverse centrality score (-1/x) to strongly prioritize the most peripheral nodes,
       which are the safest to pick first in a greedy MIS algorithm.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]
    
    # Handle edge case of empty graph
    if n == 0:
        return np.array([])
    
    # Compute degrees
    degrees = adjacency.sum(axis=1)
    
    # Initialize with degrees (normalized) for better convergence
    if np.all(degrees == 0):
        x = np.ones(n)
    else:
        x = degrees.copy()
        norm = np.linalg.norm(x)
        if norm > 0:
            x = x / norm
        else:
            x = np.ones(n) / np.sqrt(n)
            
    # Power iteration to find principal eigenvector
    # Fixed number of iterations for consistency
    for _ in range(100):
        x_new = adjacency @ x
        norm = np.linalg.norm(x_new)
        if norm < 1e-15:
            # Fallback if vector collapses (e.g., bipartite with bad init, though unlikely with degrees)
            # Keep previous x or break
            break
        x_new = x_new / norm
        x = x_new
        
    # Ensure non-negative
    x = np.abs(x)
    
    # Small epsilon to avoid division by zero for isolated nodes (x=0)
    # Isolated nodes have x=0, so 1/(0+eps) is large -> high priority.
    # This is correct: isolated nodes should be picked first.
    eps = 1e-6
    scores = -1.0 / (x + eps)
    
    return scores
# EVOLVE-BLOCK-END
