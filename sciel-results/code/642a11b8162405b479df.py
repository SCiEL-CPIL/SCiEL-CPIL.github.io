"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    """Return an edge-attractiveness matrix for ACO TSP.
    
    Uses inverse distance with a degree-balance penalty. The penalty
    discourages ants from over-concentrating on high-connectivity nodes
    by reducing the attractiveness of edges incident to nodes with many
    close neighbors. This helps prevent premature convergence to suboptimal
    tours that exploit hub-like structures.
    """
    start_time = time.time()
    
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    
    # Safety check for deadline
    if time.time() - start_time > 290:
        return 1.0 / (d + 1e-9)
    
    # Replace diagonal with large value
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e10)
    
    # Base heuristic: inverse distance
    eta_base = 1.0 / (d_safe + 1e-18)
    
    # Compute degree balance penalty
    # For each node, count how many other nodes are within its top-k nearest neighbors
    # Use a simple metric: for each node i, sum of 1/d(i,j) over all j != i
    # Nodes with high "attractiveness sum" are hubs; penalize edges to/from them
    degree_score = np.sum(eta_base, axis=1)  # shape (n,)
    
    # Normalize degree_score to [0, 1] range for penalty calculation
    d_min = np.min(degree_score)
    d_max = np.max(degree_score)
    if d_max > d_min:
        degree_norm = (degree_score - d_min) / (d_max - d_min)
    else:
        degree_norm = np.zeros(n)
    
    # Penalty factor: reduce attractiveness of edges incident to hub nodes
    # penalty[i,j] = 1 / (1 + alpha * max(degree_norm[i], degree_norm[j]))
    # Use alpha = 0.5 for moderate penalty
    alpha = 0.5
    max_degree = np.maximum(degree_norm[:, np.newaxis], degree_norm[np.newaxis, :])
    penalty = 1.0 / (1.0 + alpha * max_degree)
    
    # Apply penalty to base heuristic
    eta = eta_base * penalty
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
