"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix based on dead-gap reward:
    entry (i,j) is inversely proportional to the count of items that fit
    into the residual space capacity - (d_i + d_j).
    
    Pairs that create "dead gaps" (too small for any remaining item) are
    strongly rewarded, while pairs leaving large voids are penalized.
    """
    start_time = time.time()
    time_limit = 280.0  # Safety margin before 300s kill
    
    n = len(demands)
    demands = np.asarray(demands, dtype=float)
    
    # Pair sums and validity
    pair_sums = demands[:, None] + demands[None, :]
    valid = pair_sums <= capacity
    remaining = capacity - pair_sums
    
    # For each pair (i, j), count how many items k fit into remaining space
    # i.e., demands[k] <= remaining[i, j]
    # Vectorized: for each unique remaining value, count items <= it
    # Use searchsorted on sorted demands for efficiency
    sorted_demands = np.sort(demands)
    
    # remaining is (n, n) array; for each entry, count items <= remaining
    # np.searchsorted(sorted_demands, remaining, side='right') gives count
    counts = np.searchsorted(sorted_demands, remaining, side='right')
    
    # counts[i,j] = number of items that fit in residual space after placing i and j
    # We want: low counts (dead gaps) => high attractiveness
    # Use 1 / (counts + 1) so that count=0 (perfect or dead gap) gets 1.0
    # and larger counts get smaller values
    
    # Only for valid pairs
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # For valid pairs: score = 1.0 / (counts + 1.0)
    # This rewards pairs that leave no room for additional items
    eta[valid] = 1.0 / (counts[valid] + 1.0)
    
    # Diagonal: self-pairing is not meaningful
    np.fill_diagonal(eta, 1e-9)
    
    # Ensure all values are positive and finite
    eta[eta <= 0] = 1e-9
    eta[~np.isfinite(eta)] = 1e-9
    
    return eta
# EVOLVE-BLOCK-END
