"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin before 90s kill
    
    N = 10000000
    
    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    
    if time.time() > deadline - 10:
        return [11, 6]
    
    # Group numbers by sigma value
    groups = {}
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    # For each group with >= 2 elements, find pair minimizing |a/b - 3/2|
    best_error = 1.0  # worse than seed's 1/3
    best_pair = (11, 6)
    
    # Sort groups and process
    # Only process groups where it's possible to get error < best_error
    # |a/b - 3/2| < best_error means a/b is in (3/2 - best_error, 3/2 + best_error)
    # So a is in b*(3/2 - best_error) to b*(3/2 + best_error)
    
    # Process groups in order of size, largest first (more likely to have good pairs)
    group_items = [(len(nums), s, nums) for s, nums in groups.items() if len(nums) >= 2]
    group_items.sort(reverse=True)
    
    for size, s, nums in group_items:
        if time.time() > deadline - 2:
            break
        
        nums_sorted = sorted(nums)
        n = len(nums_sorted)
        
        # For each b, find a closest to 3b/2
        for i in range(n):
            if time.time() > deadline - 1:
                break
            b = nums_sorted[i]
            target_a = 3 * b // 2
            pos = bisect_left(nums_sorted, target_a)
            
            candidates = []
            if pos < n:
                candidates.append(nums_sorted[pos])
            if pos > 0:
                candidates.append(nums_sorted[pos - 1])
            
            for a in candidates:
                if a == b:
                    continue
                error_num = abs(2 * a - 3 * b)
                error_den = 2 * b
                error = error_num / error_den
                if error < best_error:
                    best_error = error
                    best_pair = (a, b)
    
    a, b = best_pair
    a = int(a)
    b = int(b)
    if sigma[a] != sigma[b] or a == b:
        return [11, 6]
    return [a, b]

# EVOLVE-BLOCK-END
