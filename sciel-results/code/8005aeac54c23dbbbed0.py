import numpy as np

def heuristics(demands, capacity):
    n = len(demands)
    # Sort demands for the count calculation
    sorted_demands = np.sort(demands)
    
    # Precompute the maximum number of items that fit in residual R
    # R ranges from 0 to capacity
    count_res = np.zeros(capacity + 1, dtype=np.float64)
    current_sum = 0
    count = 0
    # Two-pointer / greedy fill for each R
    # Since R increases, we can just keep adding items
    # But we need to restart for each R? No.
    # If R increases, the max count is non-decreasing.
    # We can iterate R from 0 to capacity and maintain the pointer.
    # However, the "sum of smallest items" logic is:
    # Find max k such that sum(sorted[0:k]) <= R.
    # We can precompute prefix sums.
    prefix_sums = np.cumsum(sorted_demands)
    # For each R, count is the number of prefix sums <= R
    # We can use searchsorted
    # prefix_sums is sorted.
    # searchsorted(prefix_sums, R, side='right') gives the count.
    
    for R in range(capacity + 1):
        count_res[R] = np.searchsorted(prefix_sums, R, side='right')
        
    # Compute pairwise sums
    pair_sums = demands[:, None] + demands[None, :]
    residuals = capacity - pair_sums
    
    # Mask for valid transitions
    valid = residuals >= 0
    
    # Get counts for valid residuals
    # Clip residuals to [0, capacity] for indexing
    residuals_int = np.clip(residuals.astype(int), 0, capacity)
    counts = count_res[residuals_int]
    
    # Heuristic: maximize count
    # If count is 0, it becomes 1e-9 later
    H = np.where(valid, counts, 1e-9)
    
    # Ensure positive
    H = np.where(H > 0, H, 1e-9)
    
    return H
