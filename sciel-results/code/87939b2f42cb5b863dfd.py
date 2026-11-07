"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    n = weights.shape[1]
    m = weights.shape[0]
    capacity = 25.0
    
    # Item total prizes
    item_prizes = prizes.sum(axis=0)  # shape (n,)
    
    # Compute pairwise conflict matrix
    conflict = np.zeros((n, n), dtype=np.float64)
    for c in range(m):
        pair_sum = weights[c, :][:, None] + weights[c, :][None, :]
        conflict |= (pair_sum > capacity)
    
    # Exclude self-conflict
    np.fill_diagonal(conflict, 0)
    
    # Displaced prize for each item
    displaced = conflict @ item_prizes  # shape (n,)
    
    # Competitive exclusion heuristic
    heuristic = item_prizes / (1e-9 + displaced)
    
    # Ensure positive
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
