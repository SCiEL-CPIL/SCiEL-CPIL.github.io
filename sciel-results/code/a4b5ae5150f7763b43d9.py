import time
import math
import numpy as np
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = 87.0  # Leave 3s margin for safety
    
    N = 10_000_000
    
    # 1. Compute sigma(n) for all n <= N using additive sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Vectorized additive sieve
    for i in range(1, N + 1):
        sigma[i::i] += i
        if i % 200000 == 0:
            if time.time() - start_time > deadline - 35:
                return [11, 6]
    
    if time.time() - start_time > deadline - 25:
        return [11, 6]
    
    # 2. Group numbers by their sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[int(sigma[n])].append(n)
        
    if time.time() - start_time > deadline - 10:
        return [11, 6]
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    early_exit_threshold = 1e-10
    
    # 3. Search for pairs - focus on groups with many elements first
    # Sort by group size descending to find good candidates faster
    sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
    
    # Limit to top groups by size to stay within time budget
    max_groups_to_process = 5000
    
    for group_idx, (s_val, nums) in enumerate(sorted_groups):
        if time.time() - start_time > deadline - 2:
            break
        if group_idx >= max_groups_to_process:
            break
            
        if len(nums) < 2:
            continue
            
        nums.sort()
        K = len(nums)
        
        # For each b in nums, find a in nums closest to 1.5 * b
        for b in nums:
            if time.time() - start_time > deadline - 1:
                break
                
            target = 1.5 * b
            idx = bisect.bisect_left(nums, target)
            
            # Check neighbors around the target
            for j in [idx, idx - 1, idx + 1, idx - 2]:
                if 0 <= j < K:
                    a = nums[j]
                    if a == b:
                        continue
                    if 2 * a == 3 * b:
                        continue
                    
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    err = num / den
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
                        if best_error < early_exit_threshold:
                            return [int(best_a), int(best_b)]
                        
    if best_error == float('inf'):
        return [11, 6]
        
    return [int(best_a), int(best_b)]
