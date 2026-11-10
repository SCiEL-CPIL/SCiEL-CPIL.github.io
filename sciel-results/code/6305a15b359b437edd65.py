import numpy as np
import math
import time
from collections import defaultdict
import bisect

def construct():
    start = time.time()
    deadline = start + 80  # Safety margin before 90s kill
    
    N = 10_000_000
    
    # Phase 1: Compute sigma(n) for all n using vectorized additive sieve
    # This is the strongest structural idea from inspiration 3
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
        if time.time() > deadline - 25:
            # If we're running out of time, reduce N
            N = i
            break
    
    if time.time() > deadline - 15:
        return [11, 6]  # Fallback
    
    # Phase 2: Group numbers by sigma value
    # Convert to list for faster access in Python loop
    sigma_list = sigma.tolist()
    sigma_to_numbers = defaultdict(list)
    for n in range(1, N + 1):
        sigma_to_numbers[sigma_list[n]].append(n)
    
    if time.time() > deadline - 10:
        return [11, 6]  # Fallback
    
    # Phase 3: Search for best pair
    # Error = |a/b - 3/2| = |2a - 3b| / (2b)
    # We want to minimize this
    # For each sigma group, sort the numbers
    # For each b in the group, find a in the group closest to 3b/2
    
    best_a, best_b = 11, 6
    # Initial error: |2*11 - 3*6| / (2*6) = |22-18|/12 = 4/12 = 1/3
    best_num = 4   # numerator of error
    best_den = 12  # denominator of error
    
    # Process groups
    for s, nums in sigma_to_numbers.items():
        if time.time() > deadline - 5:
            break
        if len(nums) < 2:
            continue
        
        # Sort the numbers in this group
        nums_sorted = sorted(nums)
        nlen = len(nums_sorted)
        
        # For each b, find a closest to 3b/2
        for idx_b in range(nlen):
            if time.time() > deadline - 3:
                break
            
            b = nums_sorted[idx_b]
            target_a = 3 * b / 2.0
            
            # Binary search for closest to target_a
            pos = bisect.bisect_left(nums_sorted, target_a)
            
            # Check pos and pos-1
            for idx_a in (pos, pos - 1):
                if idx_a < 0 or idx_a >= nlen:
                    continue
                a = nums_sorted[idx_a]
                if a == b:
                    continue
                
                # Compute error: |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                den = 2 * b
                
                # Compare num/den vs best_num/best_den
                # num * best_den < best_num * den
                if num * best_den < best_num * den:
                    best_num = num
                    best_den = den
                    best_a, best_b = a, b
    
    return [int(best_a), int(best_b)]
