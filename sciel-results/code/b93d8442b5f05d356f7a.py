import time
import numpy as np
from bisect import bisect_left
from collections import defaultdict

def construct():
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin (90s limit)
    
    MAX_N = 10_000_000
    
    # 1. Compute sigma using NumPy
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d
        # Time check inside sieve?
        # The sieve loop is ~10M iterations.
        # NumPy slicing is fast.
        # 10M iterations in Python loop for d is ~1s.
        # The inner `sigma[d::d] += d` is O(N/d).
        # Total ops N log N.
        # Should be < 5 seconds.
        
    if time.time() > deadline - 20:
        # Fallback if sieve is too slow (unlikely with NumPy)
        return [11, 6]

    # 2. Group numbers by sigma value
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        s = int(sigma[n])
        groups[s].append(n)
        
    if time.time() > deadline - 10:
        return [11, 6]

    # 3. Search for best pair
    best_err = float('inf')
    best_pair = (11, 6) # Fallback
    
    # Process larger groups first (more likely to have close ratios)
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
    
    for s, nums in sorted_groups:
        if time.time() > deadline - 3:
            break
        if len(nums) < 2:
            continue
            
        n = len(nums)
        for i in range(n):
            b = nums[i]
            target = 1.5 * b
            pos = bisect_left(nums, target)
            
            # Check a wider window for better precision
            start = max(0, pos - 10)
            end = min(n, pos + 11)
            
            for j in range(start, end):
                if j == i:
                    continue
                a = nums[j]
                # Exact rational: |a/b - 3/2| = |2a - 3b| / (2b)
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    continue
                err = diff / (2.0 * b)
                if err < best_err:
                    best_err = err
                    best_pair = (a, b)
                    
    a, b = best_pair
    return [int(a), int(b)]
