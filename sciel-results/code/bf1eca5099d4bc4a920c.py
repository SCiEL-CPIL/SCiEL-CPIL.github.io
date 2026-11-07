import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes a static heuristic matrix for ACO based on 'Effective Cost'.
    The heuristic penalizes edges that lead to nodes which are expensive to reach 
    (high d[i,j]) AND expensive to return from (high d[j,0]).
    This models the opportunity cost of the path length budget.
    """
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    # Ensure shapes are correct
    n = len(prizes)
    if distance_matrix.shape != (n, n):
        raise ValueError("Distance matrix shape mismatch")
        
    # Distance from each node j to the depot (node 0)
    d_j0 = distance_matrix[:, 0]
    
    # Effective cost of moving i -> j:
    # We pay d[i,j] to get there, and we will likely pay d[j,0] to return eventually.
    # Summing them gives a proxy for the total budget impact of including j in the path from i.
    # Note: This is an approximation, as d[j,0] is paid only once, not per edge.
    # But it correctly identifies nodes that are "far out" as less attractive.
    
    # Add epsilon to avoid division by zero
    eps = 1e-9
    
    # Compute effective cost matrix
    # d[i,j] is distance_matrix[i,j]
    # d[j,0] is d_j0[j] (broadcasted along rows)
    effective_cost = distance_matrix + d_j0[None, :] + eps
    
    # Heuristic: Prize / (Effective Cost)^exponent
    # Exponent 2.0 emphasizes efficiency (short paths) more strongly.
    # Let's try 2.0.
    exponent = 2.0
    
    heuristic_matrix = prizes[None, :] / (effective_cost ** exponent)
    
    # Handle non-positive values
    heuristic_matrix[heuristic_matrix <= 0] = 1e-9
    
    # Ensure finite values
    heuristic_matrix[~np.isfinite(heuristic_matrix)] = 1e-9
    
    return heuristic_matrix
