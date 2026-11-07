import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    d = np.asarray(demands, dtype=np.float64)
    
    # Pair sums
    pair_sum = d[:, None] + d[None, :]
    
    # Residual space for each pair
    residual = capacity - pair_sum
    
    # Initialize heuristic
    heuristic = np.full((n, n), 1e-9, dtype=np.float64)
    
    # For pairs that fit
    fits = pair_sum <= capacity
    
    # For each fitting pair, count how many other items fit in the residual
    # We want to count items k (k != i, k != j) where d[k] <= residual[i,j]
    
    # Efficient approach: for each pair (i,j) that fits, count items with d <= residual
    # This is O(n^3) naive, but n=120 so it's fine
    
    # Precompute sorted demands for binary search
    sorted_d = np.sort(d)
    
    for i in range(n):
        for j in range(n):
            if not fits[i, j]:
                continue
            r = residual[i, j]
            # Count items k where d[k] <= r, excluding i and j
            # Number of items with d <= r
            count_all = np.searchsorted(sorted_d, r, side='right')
            # Subtract i and j if they themselves fit in residual r
            # But i and j are already placed, so we exclude them
            # Check if d[i] <= r
            if d[i] <= r:
                count_all -= 1
            if d[j] <= r:
                count_all -= 1
            # count_all is now the number of other items that fit
            heuristic[i, j] = 1.0 / (count_all + 1)
    
    return heuristic
