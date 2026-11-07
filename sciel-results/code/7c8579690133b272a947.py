import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    N = 10_000_000
    
    # 1. Compute sigma(n) using NumPy sieve (vectorized)
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Check time before starting heavy computation
    if time.time() - start_time > deadline - 15:
        return [11, 6]
    
    for i in range(1, N + 1):
        sigma[i::i] += i
    
    if time.time() - start_time > deadline - 5:
        return [11, 6]
    
    # 2. Build a mapping from sigma value to list of numbers
    # Only keep numbers that could be relevant (a ≈ 1.5*b)
    sigma_to_nums = defaultdict(list)
    
    for n in range(1, N + 1):
        s = int(sigma[n])
        sigma_to_nums[s].append(n)
    
    if time.time() - start_time > deadline - 3:
        return [11, 6]
    
    # 3. Search for pairs where a/b ≈ 3/2 and sigma(a) = sigma(b)
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # For each sigma value, check all pairs (a,b) in that group
    # where a/b is close to 3/2
    for s_val, nums in sigma_to_nums.items():
        if time.time() - start_time > deadline:
            break
        
        if len(nums) < 2:
            continue
        
        # Sort nums for binary search
        nums_sorted = sorted(nums)
        
        for b in nums_sorted:
            target = 1.5 * b
            # Find a values close to target using binary search
            # We want |a/b - 3/2| = |2a - 3b| / (2b) to be small
            # So we want 2a ≈ 3b, i.e., a ≈ 1.5b
            
            # Check a few candidates around target
            a_lo = int(math.floor(target)) - 2
            a_hi = int(math.ceil(target)) + 2
            
            for a in range(max(1, a_lo), min(N, a_hi) + 1):
                if a == b:
                    continue
                if 2 * a == 3 * b:
                    continue  # Exact 3/2 is invalid
                if a in sigma_to_nums[s_val]:
                    numerator = abs(2 * a - 3 * b)
                    denominator = 2 * b
                    err = numerator / denominator
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
    
    return [int(best_a), int(best_b)]
