import time
import numpy as np
from bisect import bisect_left
from collections import defaultdict

def construct():
    t_start = time.time()
    deadline = t_start + 80.0  # Safety margin
    
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
        
    if time.time() > deadline - 10:
        # Fallback if sieve is too slow (unlikely with NumPy)
        return [11, 6]

    # 2. Group numbers by sigma value
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        s = int(sigma[n])
        groups[s].append(n)
        
    if time.time() > deadline - 5:
        return [11, 6]

    # 3. Search for best pair
    best_err = float('inf')
    best_pair = (11, 6) # Fallback
    
    # We can optimize by skipping groups with size 1
    # Also, we can break early if we find error 0? No, error must be > 0.
    # We can break if best_err is extremely small? No, we want the minimum.
    
    for s, nums in groups.items():
        if time.time() > deadline - 2:
            break
        if len(nums) < 2:
            continue
            
        # nums is already sorted because we iterated n from 1 to MAX_N
        # For each b in nums, find a in nums closest to 1.5 * b
        for b in nums:
            target = 1.5 * b
            pos = bisect_left(nums, target)
            
            # Check pos
            if pos < len(nums):
                a = nums[pos]
                if a != b:
                    err = abs(a / b - 1.5)
                    if 0 < err < best_err:
                        best_err = err
                        best_pair = (a, b)
            
            # Check pos - 1
            if pos > 0:
                a = nums[pos - 1]
                if a != b:
                    err = abs(a / b - 1.5)
                    if 0 < err < best_err:
                        best_err = err
                        best_pair = (a, b)
                        
    a, b = best_pair
    return [int(a), int(b)]
