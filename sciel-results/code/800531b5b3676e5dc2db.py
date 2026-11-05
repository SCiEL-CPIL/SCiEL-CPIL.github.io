import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145  # Safety margin

    n = adjacency.shape[0]
    adj = np.array(adjacency, dtype=np.float64)
    
    # Compute degrees
    degrees = adj.sum(axis=1)
    
    # Compute eigenvector centrality using power iteration
    # This is fast and stable for sparse-ish graphs
    # We want the principal eigenvector of the adjacency matrix
    # Low centrality vertices are often "peripheral" and good candidates for MIS
    
    # Initialize with small random or uniform values
    np.random.seed(42)  # Fixed seed for reproducibility
    vec = np.random.rand(n)
    vec = vec / np.linalg.norm(vec)
    
    # Power iteration
    for _ in range(50):  # 50 iterations is usually enough for convergence
        new_vec = adj @ vec
        norm = np.linalg.norm(new_vec)
        if norm < 1e-10:
            break
        vec = new_vec / norm
    
    # Combine degree and centrality
    # Strategy: Prefer low degree and low centrality vertices
    # Normalize both to [0, 1]
    deg_min, deg_max = degrees.min(), degrees.max()
    if deg_max > deg_min:
        deg_norm = (degrees - deg_min) / (deg_max - deg_min)
    else:
        deg_norm = np.zeros(n)
    
    cen_min, cen_max = vec.min(), vec.max()
    if cen_max > cen_min:
        cen_norm = (vec - cen_min) / (cen_max - cen_min)
    else:
        cen_norm = np.zeros(n)
    
    # Score: We want vertices with LOW degree and LOW centrality to have HIGH priority
    # So we invert both and combine
    # Higher score = higher priority in greedy decoder
    score = (1.0 - deg_norm) * 0.5 + (1.0 - cen_norm) * 0.5
    
    # Add a small random perturbation to break ties
    score += np.random.rand(n) * 1e-6
    
    # Ensure scores are unique by adding tiny increments
    order = np.argsort(-score)
    for i, v in enumerate(order):
        score[v] += i * 1e-9
    
    return score
