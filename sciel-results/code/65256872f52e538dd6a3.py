"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    # demands is a 1D array of floats
    # Create a 2D array of sums for all pairs (i, j)
    # pair[i, j] = demands[i] + demands[j]
    pair = demands[:, None] + demands[None, :]
    
    # Calculate remaining capacity for each pair
    # rem[i, j] = capacity - (demands[i] + demands[j])
    rem = capacity - pair
    
    # We only care about pairs that fit (rem >= 0)
    valid = rem >= 0
    
    # Initialize heuristic matrix with 1e-9 (invalid/default)
    heur = np.full((n, n), 1e-9, dtype=np.float64)
    
    # New Approach: Pure Decreasing Heuristic (Greedy Largest Fit)
    # The previous heuristic assumed the last item 'i' represented the bin's state,
    # which is incorrect for bins with >2 items. This new heuristic ignores the
    # specific 'i' for the value and only uses it for validity, focusing entirely
    # on placing the largest possible item 'j' next. This mimics First Fit Decreasing.
    
    # demands_j is shape (1, n), broadcasts to (n, n)
    demands_j = demands[None, :]
    
    # Set heuristic to demand size for valid pairs
    # We need to ensure we only update valid entries
    # Using boolean indexing
    rows, cols = np.where(valid)
    heur[rows, cols] = demands[cols]
    
    # Ensure no non-positive values remain (demands are positive, so this is fine)
    # But just in case, clip
    heur = np.maximum(heur, 1e-9)
    
    return heur
# EVOLVE-BLOCK-END
