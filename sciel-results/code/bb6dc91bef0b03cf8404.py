import numpy as np
import time

def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    Uses a power-law heuristic: prize^alpha / cost^beta.
    """
    n = len(prizes)
    d = distance_matrix
    d_to_0 = d[:, 0]
    
    # Effective cost: distance from i to j plus distance from j to 0
    C = d + d_to_0[None, :]
    
    # Ensure costs are positive to avoid division by zero
    C_safe = np.maximum(C, 1e-6)
    
    # Parameters for the power-law heuristic
    # Higher alpha emphasizes high prizes more strongly
    # Lower beta reduces the penalty for long distances
    alpha = 2.0
    beta = 1.0
    
    # Calculate heuristic
    # prizes shape (n,), C_safe shape (n, n)
    # Result shape (n, n)
    heuristic = (prizes[None, :] ** alpha) / (C_safe ** beta)
    
    # Handle edge cases: non-finite or non-positive values
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
