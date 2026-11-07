"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Cost of visiting j from i and returning to 0
    # d[i, j] + d[j, 0]
    # This represents the marginal cost of adding j to the path.
    
    # We want to encourage moves that have high prize and low marginal cost.
    
    # Initialize heuristic matrix
    heuristic = np.zeros((n, n), dtype=float)
    
    # Calculate marginal costs for all i, j
    # d_ij is distance_matrix
    # d_j0 is distance_matrix[:, 0] (column 0)
    
    # Add epsilon to avoid division by zero (though distances are > 0 for distinct nodes usually)
    eps = 1e-6
    
    # Marginal cost matrix: C[i, j] = d[i, j] + d[j, 0]
    # d[j, 0] is the distance from node j to node 0.
    # distance_matrix is n x n.
    # d[j, 0] for all j is distance_matrix[:, 0]
    
    d_j0 = distance_matrix[:, 0]
    
    # C[i, j] = distance_matrix[i, j] + d_j0[j]
    # This is (n, n) + (n,) -> broadcast to (n, n)
    marginal_cost = distance_matrix + d_j0[None, :]
    
    # Heuristic formula: prize_j^alpha / (marginal_cost + eps)^beta
    # alpha = 2.0 to emphasize high prizes
    # beta = 1.5 to penalize high cost
    alpha = 2.0
    beta = 1.5
    
    prize_term = np.power(prizes[None, :], alpha)
    cost_term = np.power(marginal_cost + eps, beta)
    
    heuristic = prize_term / cost_term
    
    # Set diagonal to 0 (or very small) as we don't move to self
    # And ensure non-positive values are 1e-9 as per contract
    np.fill_diagonal(heuristic, 1e-9)
    
    # Handle any 0 or negative values (shouldn't happen with +eps, but safe)
    heuristic = np.where(heuristic > 0, heuristic, 1e-9)
    
    # Ensure finite
    heuristic = np.where(np.isfinite(heuristic), heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
