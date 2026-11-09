import numpy as np

def priority(adjacency):
    # Ensure adjacency is a numeric matrix
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    
    # Compute degrees
    degrees = A.sum(axis=1)
    
    # Compute sum of neighbor degrees
    # S[v] = sum of degrees of neighbors of v
    # S = A @ degrees
    neighbor_degree_sums = A @ degrees
    
    # Initialize scores
    # Base score: -degree
    scores = -degrees.copy()
    
    # Boost by mean neighbor degree
    # For vertices with degree > 0:
    # mean_neighbor_degree = neighbor_degree_sums / degrees
    # Score += beta * mean_neighbor_degree
    
    beta = 1.0
    mask = degrees > 0
    if np.any(mask):
        mean_neighbor_degrees = neighbor_degree_sums[mask] / degrees[mask]
        scores[mask] += beta * mean_neighbor_degrees
    
    # For isolated vertices (degree 0), assign a very high score
    # to ensure they are picked first
    scores[~mask] = 1e6
    
    return scores
