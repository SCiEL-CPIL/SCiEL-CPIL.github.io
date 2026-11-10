import time
import sys
import numpy as np
from bisect import bisect_left

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for n <= N using NumPy vectorized sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for i in range(1, N + 1):
        sigma[i::i] += i
        if time.time() > deadline - 10.0:
            # Fallback to known valid pair
            return [11, 6]
    
    if time.time() > deadline - 5.0:
        return [11, 6]
    
    # 2. Group numbers by sigma value
    # Use a dict mapping sigma -> list of n
    groups = {}
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]
    
    if time.time() > deadline - 2.0:
        return [11, 6]
    
    # 3. Find best pair: minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For exact comparison: minimize |2a-3b| * b' vs |2a'-3b'| * b
    best_num = None  # numerator of the fraction |2a-3b|
    best_den = None  # denominator 2b
    best_a = 11
    best_b = 6
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        
        nums.sort()
        nlen = len(nums)
        
        for idx_b in range(nlen):
            b = nums[idx_b]
            target_a = 1.5 * b
            pos = bisect_left(nums, target_a)
            
            # Check pos and pos-1
            candidates = []
            if pos < nlen:
                candidates.append(nums[pos])
            if pos > 0:
                candidates.append(nums[pos-1])
            
            for a in candidates:
                diff = abs(2 * a - 3 * b)
                if diff == 0:
                    # a/b = 3/2 exactly, but this is invalid per rules
                    continue
                # Compare diff/(2b) with best_num/best_den
                # diff/(2b) < best_num/best_den  iff  diff * best_den < best_num * 2b
                if best_num is None:
                    best_num = diff
                    best_den = 2 * b
                    best_a = a
                    best_b = b
                else:
                    lhs = diff * best_den
                    rhs = best_num * 2 * b
                    if lhs < rhs:
                        best_num = diff
                        best_den = 2 * b
                        best_a = a
                        best_b = b
        
        if time.time() > deadline - 1.0:
            break
    
    if best_num is None:
        return [11, 6]
    
    return [int(best_a), int(best_b)]
# EVOLVE-BLOCK-END
