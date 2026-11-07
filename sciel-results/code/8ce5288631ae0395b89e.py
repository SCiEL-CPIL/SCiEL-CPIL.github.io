import sys
import time
import math

# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict

MAX_N = 10000000

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # Safety margin before 90s kill
    
    # 1. Compute sigma(n) for all n <= MAX_N using an efficient sieve
    # Use a linear-style approach: for each d, add d to multiples
    # But vectorize with NumPy in chunks to avoid Python loop overhead
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Process divisors in batches using vectorized operations
    # For small d, the number of multiples is large; for large d, it's small
    # We'll do it in one pass but check time periodically
    
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d
        
        if d % 500000 == 0:
            if time.time() > deadline - 10.0:
                # Not enough time to finish sieve + search
                # Return best known pair
                return [11, 6]
    
    if time.time() > deadline - 5.0:
        return [11, 6]
    
    # 2. Build groups: map sigma value -> list of n's
    # This allows O(1) lookup of all n with a given sigma
    sig_list = sigma.tolist()
    
    # Build dictionary: sigma_value -> sorted list of indices
    # Only for values that appear at least twice (to save memory/time)
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sv = sig_list[n]
        groups[sv].append(n)
    
    # Filter to only groups with >= 2 elements
    valid_groups = {sv: nums for sv, nums in groups.items() if len(nums) >= 2}
    
    # 3. Search for best pair
    # Minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For each group, we want a/b close to 3/2
    # i.e., 2a close to 3b
    
    best_a, best_b = 11, 6
    best_err_num, best_err_den = 4, 12  # |11/6 - 3/2| = |11-9|/6 = 2/6 = 1/3
    best_score_val = 1.0 / 3.0
    
    # For efficiency, iterate over groups and for each pair (a,b) in the group
    # check if sigma(a) == sigma(b) (guaranteed by grouping) and compute error
    # But groups can be large, so we need to be smart
    
    # Strategy: for each group, sort by value. For each b, the best a is near 1.5*b
    # Use binary search to find a near 1.5*b
    
    import bisect
    
    for sv, nums in valid_groups.items():
        if time.time() > deadline - 2.0:
            break
        
        nums_sorted = sorted(nums)  # already sorted from construction, but ensure
        
        # For each b in the group, find a near 1.5*b
        # To limit work, only check b values that could improve
        for b in nums_sorted:
            # Pruning: min possible error with this b is 1/(2b) if 3b is odd, 
            # or 2/(2b)=1/b if 3b is even
            # If 1/(2b) >= best_score_val, skip
            if 1.0 / (2.0 * b) >= best_score_val:
                continue
            
            target = 1.5 * b
            # Find position in nums_sorted
            idx = bisect.bisect_left(nums_sorted, target)
            
            # Check a few candidates around idx
            for i in range(max(0, idx - 5), min(len(nums_sorted), idx + 6)):
                a = nums_sorted[i]
                if a == b:
                    continue
                # Compute error |a/b - 3/2| = |2a - 3b| / (2b)
                k = abs(2 * a - 3 * b)
                if k == 0:
                    continue  # invalid: exact ratio
                # Compare k/(2b) with best_err_num/best_err_den
                # k/(2b) < best_err_num/best_err_den  iff  k * best_err_den < best_err_num * 2 * b
                if k * best_err_den < best_err_num * 2 * b:
                    best_a, best_b = a, b
                    best_err_num, best_err_den = k, 2 * b
                    best_score_val = k / (2.0 * b)
    
    return [int(best_a), int(best_b)]

# EVOLVE-BLOCK-END
