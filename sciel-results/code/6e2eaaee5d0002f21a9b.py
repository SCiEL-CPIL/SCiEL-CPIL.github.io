"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    if n == 0:
        return np.array([])
    
    # Distance from each node to the depot (node 0)
    d_to_depot = distance_matrix[:, 0]
    
    # Marginal cost of moving from i to j and then returning to depot
    # Cost(i->j->0) - Cost(i->0) = d[i,j] + d[j,0] - d[i,0]
    # This represents the extra length incurred by visiting j before returning.
    
    # d[i,j] is distance_matrix[i, j]
    # d[j,0] is d_to_depot[j] -> shape (1, n) for broadcasting
    # d[i,0] is d_to_depot[i] -> shape (n, 1) for broadcasting
    
    d_j0_row = d_to_depot[None, :]
    d_i0_col = d_to_depot[:, None]
    
    marginal_cost = distance_matrix + d_j0_row - d_i0_col
    
    # Ensure marginal cost is positive to avoid division by zero or negative
    # Triangle inequality ensures this is >= 0, but add epsilon for safety
    eps = 1e-9
    marginal_cost = np.maximum(marginal_cost, eps)
    
    # Heuristic: Prize divided by marginal cost
    # prizes shape (n,) -> (n, 1)
    prizes_col = prizes[:, None]
    
    H = prizes_col / marginal_cost
    
    # Clean up non-finite and non-positive values
    H[~np.isfinite(H)] = 1e-9
    H[H <= 0] = 1e-9
    
    return H
# EVOLVE-BLOCK-END
