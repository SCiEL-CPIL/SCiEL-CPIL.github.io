"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(demands, capacity):
    """
    Computes a heuristic matrix based on a 'dead gap' reward.
    
    Strategy:
    1. For each valid pair (i, j) where d_i + d_j <= capacity, calculate the residual space:
       residual = capacity - (d_i + d_j)
    2. Count how many other items k (k != i, k != j) can fit into this residual space.
       Let count_k be the number of such items.
    3. The attractiveness eta[i, j] is inversely proportional to (1 + count_k).
       - If count_k is 0 (dead gap, no other item fits), eta is maximal.
       - If count_k is large (lots of flexibility), eta is lower.
       
    This rewards pairs that pack bins tightly, leaving gaps too small for any remaining item,
    which generally leads to fewer bins needed.
    """
    n = len(demands)
    demands = np.asarray(demands, dtype=float)
    
    # Create the heuristic matrix
    eta = np.full((n, n), 1e-9, dtype=float)
    
    # Compute pair sums
    pair_sums = demands[:, None] + demands[None, :]
    
    # Valid mask: pairs that fit in capacity
    valid_mask = pair_sums <= capacity
    
    # Compute residual for each pair
    residuals = capacity - pair_sums
    
    # For each pair (i, j), we want to count how many items k fit in residuals[i, j]
    # This is a 3D problem if done naively, but we can vectorize per pair or use a trick.
    
    # Efficient approach:
    # For a given residual r, the number of items k (excluding i and j) that fit is:
    # sum(1 for k in range(n) if k != i and k != j and demands[k] <= r)
    
    # We can precompute a sorted list of demands to quickly count how many items <= r.
    # Let sorted_demands be the sorted demands.
    # Let counts_less_equal[r] be the count of items <= r.
    # Since demands are integers (as floats), we can use searchsorted.
    
    sorted_demands = np.sort(demands)
    
    # For each valid pair (i, j), calculate count_k
    # We need to be careful to exclude i and j from the count.
    # count_k = (number of items <= residual[i, j]) - (1 if demands[i] <= residual[i, j] else 0) - (1 if demands[j] <= residual[i, j] else 0)
    # Note: demands[i] <= residual[i, j] is equivalent to d_i <= capacity - d_i - d_j => 2*d_i + d_j <= capacity
    # This is not always true, so we must check.
    
    # Vectorized calculation:
    # 1. Count of all items <= residual[i, j]
    #    Use np.searchsorted(sorted_demands, residuals, side='right')
    #    This gives an array of shape (n, n) with counts for all pairs.
    
    counts_all = np.searchsorted(sorted_demands, residuals, side='right')
    
    # 2. Subtract 1 if demands[i] fits in residual[i, j]
    #    Condition: demands[i] <= residuals[i, j]
    #    This is a matrix: demands[:, None] <= residuals
    fits_i = demands[:, None] <= residuals
    counts_all -= fits_i.astype(int)
    
    # 3. Subtract 1 if demands[j] fits in residual[i, j]
    #    Condition: demands[j] <= residuals[i, j]
    #    This is a matrix: demands[None, :] <= residuals
    fits_j = demands[None, :] <= residuals
    counts_all -= fits_j.astype(int)
    
    # Now counts_all[i, j] is the number of items k (k != i, k != j) that fit in the residual of pair (i, j).
    # Clamp to 0 just in case of numerical issues or logic errors (though it should be >= 0)
    counts_all = np.maximum(counts_all, 0)
    
    # Calculate attractiveness: inversely proportional to (1 + count_k)
    # eta[i, j] = 1.0 / (1.0 + counts_all[i, j])
    
    # Apply only to valid pairs
    eta[valid_mask] = 1.0 / (1.0 + counts_all[valid_mask])
    
    # Ensure diagonal is low (though valid_mask[i, i] is false if 2*d_i > capacity, 
    # but if 2*d_i <= capacity, it's valid. We want diagonal to be 1e-9 to avoid self-selection)
    np.fill_diagonal(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
