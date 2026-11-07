"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill
    
    N = 10000000
    
    # Compute sigma(n) for all n up to N using sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for i in range(1, N + 1):
        sigma[i::i] += i
    
    if time.time() > deadline - 15:
        return [11, 6]
    
    # Group numbers by sigma value using numpy for speed
    sigmas = sigma[1:]  # sigma[1] to sigma[N]
    nums = np.arange(1, N + 1)
    
    # Sort by sigma value
    order = np.argsort(sigmas, kind='stable')
    sorted_sigmas = sigmas[order]
    sorted_nums = nums[order]
    
    # Find groups (consecutive elements with same sigma)
    if time.time() > deadline - 10:
        return [11, 6]
    
    # Find boundaries where sigma changes
    changes = np.where(np.diff(sorted_sigmas) != 0)[0] + 1
    boundaries = np.concatenate([[0], changes, [N]])
    
    # Build groups as lists of arrays
    groups = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start >= 2:
            groups.append(sorted_nums[start:end])
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    # For each group with >= 2 elements, find pair minimizing |a/b - 3/2|
    best_error = 1.0  # worse than seed's 1/3
    best_pair = (11, 6)
    
    # Prioritize groups: prefer groups with large range (max-min) as they're more likely
    # to contain pairs with ratio close to 3/2
    def group_priority(g):
        if len(g) < 2:
            return 0
        return (g[-1] - g[0]) * len(g)
    
    groups.sort(key=group_priority, reverse=True)
    
    for nums_arr in groups:
        if time.time() > deadline - 3:
            break
        
        n = len(nums_arr)
        
        # Convert to list for bisect operations
        nums_list = nums_arr.tolist()
        
        for i in range(n):
            if time.time() > deadline - 2:
                break
            b = nums_list[i]
            target_a = 3 * b / 2
            pos = bisect_left(nums_list, target_a)
            
            # Check more candidates around pos for better precision
            for offset in range(0, 15):
                for p in [pos + offset, pos - 1 - offset]:
                    if 0 <= p < n:
                        a = nums_list[p]
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
