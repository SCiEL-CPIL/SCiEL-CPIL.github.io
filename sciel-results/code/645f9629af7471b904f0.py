import numpy as np
import math
import time

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors of 5040
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Filter divisors <= target and sort
    divisors = [d for d in divisors if d <= target]
    divisors.sort()
    
    # Initialize DP arrays
    # dp[s] = min number of divisors to sum to s
    INF = 10**9
    dp = np.full(target + 1, INF, dtype=np.int32)
    dp[0] = 0
    
    # parent[s] = divisor that was added to reach s optimally
    # Using int32 for parent as well, 0 means None/not set
    parent = np.zeros(target + 1, dtype=np.int32)
    
    # Process each divisor
    for d in divisors:
        # We need to update dp[d:] based on dp[:-d]
        # But we must use the state BEFORE adding d to dp[:-d]
        # So we operate on slices
        
        # Potential improvement: if d is very small, this is many operations.
        # NumPy vectorization handles this.
        
        # Calculate potential new values
        # prev_dp_slice is dp[0 : target-d+1]
        # curr_dp_slice is dp[d : target+1]
        
        # We want: new_val = old_dp[s-d] + 1
        # If new_val < old_dp[s], update
        
        # To avoid race conditions, we conceptually use the 'old' dp for the source.
        # Since we are updating dp in place, we must be careful.
        # However, if we iterate d in any order, and we do:
        # dp[d:] = np.minimum(dp[d:], dp[:-d] + 1)
        # This is WRONG for 0/1 knapsack if done in place with vectorization 
        # because dp[:-d] might have already been updated for larger s?
        # No, dp[:-d] covers indices 0 to target-d.
        # dp[d:] covers indices d to target.
        # The source indices are 0..target-d.
        # The destination indices are d..target.
        # There is an overlap if d < target-d? i.e., 2d < target.
        # If we update dp[d] using dp[0], that's fine.
        # If we update dp[2d] using dp[d], and dp[d] was just updated using dp[0], 
        # then we are using d twice.
        
        # Therefore, we MUST use a copy of the relevant part of dp for the source.
        
        # Source: dp[0 : target-d+1]
        # We need a snapshot of this.
        
        # Optimization: Only copy if necessary? 
        # For 0/1 knapsack, the standard trick is backward iteration in loops.
        # In NumPy, we can't easily do "backward vectorized iteration" without a loop.
        # So we must copy the source slice.
        
        source_slice = dp[:target - d + 1].copy()
        
        # Calculate candidate values
        candidates = source_slice + 1
        
        # Determine where candidates are better
        mask = candidates < dp[d:]
        
        # Update dp
        dp[d:] = np.where(mask, candidates, dp[d:])
        
        # Update parent
        # If mask is true, parent[d:] should be d
        # We need to set parent[d:] to d where mask is true.
        # np.where returns an array. 
        # parent[d:] = np.where(mask, d, parent[d:])
        
        # This creates a new array for the slice.
        new_parent_slice = np.where(mask, d, parent[d:])
        parent[d:] = new_parent_slice
        
    # Check if solution exists
    if dp[target] >= INF:
        return []
        
    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        d_val = parent[s]
        if d_val == 0:
            # Should not happen if dp[target] is valid
            break
        chosen.append(int(d_val))
        s -= int(d_val)
        
    return chosen
