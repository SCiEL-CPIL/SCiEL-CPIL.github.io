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
    # For pairs that don't fit, we set heuristic to 1e-9
    # The task says: "Heuristic entries that are not positive become 1e-9"
    
    # New Approach: "Large items to loose bins" (Worst Fit bias for large items)
    # Hypothesis: Placing large items into bins with more remaining space prevents
    # fragmentation where large items are left unplaced because only small spaces remain.
    # Small items can then fill the tight gaps.
    # Heuristic = demands[j] * (rem[i, j] + 1)
    # This favors:
    # 1. Large items j (demands[j] is large)
    # 2. Loose fits (rem[i, j] is large)
    
    # Initialize heuristic matrix with 1e-9 (invalid/default)
    heur = np.full((n, n), 1e-9, dtype=np.float64)
    
    # Calculate heuristic for valid pairs
    valid = rem >= 0
    
    # Extract indices of valid pairs
    rows, cols = np.where(valid)
    
    # Calculate values for valid pairs
    # demands[cols] gets the size of item j for each valid pair
    # rem[rows, cols] gets the remaining space
    heur[rows, cols] = demands[cols] * (rem[rows, cols] + 1.0)
    
    # Ensure no non-positive values remain (though calculation yields positive for valid pairs)
    # The task says "Heuristic entries that are not positive become 1e-9".
    # Our calculation yields positive values for valid pairs.
    # Invalid pairs are already 1e-9.
    
    return heur
# EVOLVE-BLOCK-END
