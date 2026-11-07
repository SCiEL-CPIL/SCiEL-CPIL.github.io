"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    # Extract distances from node 0 to all other nodes
    d_from_0 = distance_matrix[0, :]
    
    # Initialize heuristic matrix
    H = np.zeros((n, n))
    
    # Precompute the "cost to return" term for each j
    # cost_return[j] = d[j, 0]
    d_to_0 = distance_matrix[:, 0]
    
    # We iterate over all pairs (i, j)
    # Vectorized approach:
    # d_ij is distance_matrix
    # d_0i is d_from_0[i] for each row i
    # d_j0 is d_to_0[j] for each column j
    
    # Create matrices for broadcasting
    # d_0i_matrix: shape (n, 1), each row i has d[0, i]
    d_0i_matrix = d_from_0[:, np.newaxis]
    # d_j0_matrix: shape (1, n), each col j has d[j, 0]
    d_j0_matrix = d_to_0[np.newaxis, :]
    
    # d_ij is distance_matrix (n, n)
    
    # Calculate slack: max_len - d_0i - d_ij - d_j0
    # Note: d_0i is the lower bound on path length to i.
    slack = max_len - d_0i_matrix - distance_matrix - d_j0_matrix
    
    # Heuristic formula: (prize_j / d_ij) * slack
    # We add epsilon to avoid division by zero and to handle negative slack (which will be clipped to 1e-9 later by evaluator, 
    # but we can also just let it be negative here).
    # The problem states: "Heuristic entries that are not positive become 1e-9".
    # So we just need to compute the raw value.
    
    # prize_j is prizes[j], shape (n,) -> (1, n)
    prize_matrix = prizes[np.newaxis, :]
    
    # d_ij is distance_matrix
    # Add epsilon to d_ij to avoid div by zero
    d_ij_safe = distance_matrix + 1e-9
    
    # Calculate base ratio
    ratio = prize_matrix / d_ij_safe
    
    # Calculate final heuristic
    # We multiply by slack. If slack is negative, the value is negative.
    H = ratio * slack
    
    # Ensure shape is correct
    return H
# EVOLVE-BLOCK-END
