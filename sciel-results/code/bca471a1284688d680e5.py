import time
import math
import numpy as np
import bisect

def construct():
    start_time = time.time()
    deadline = 85.0  # Safety margin for 90s limit
    
    L = 10_000_000
    
    # 1. Compute sigma(n) for n=1..L
    sigma = np.zeros(L + 1, dtype=np.int64)
    for i in range(1, L + 1):
        sigma[i::i] += i
        
    # 2. Build mapping from sigma value to list of n
    sigma_map = {}
    for n in range(1, L + 1):
        s = int(sigma[n])
        if s not in sigma_map:
            sigma_map[s] = []
        sigma_map[s].append(n)
        
    # 3. Sort lists for binary search
    for s in sigma_map:
        sigma_map[s].sort()
        
    # 4. Search for best pair using fiber-based approach
    best_error = float('inf')
    best_a = 1
    best_b = 1
    
    # For each fiber with >= 2 elements, search for pairs (a,b) with a/b ≈ 3/2
    # Process fibers in order of largest max element first (to find large b early)
    # Actually, just iterate through all fibers
    
    for s, lst in sigma_map.items():
        if time.time() - start_time > deadline - 2:
            break
        if len(lst) < 2:
            continue
        
        # Iterate b from largest to smallest in this fiber
        for bi in range(len(lst) - 1, -1, -1):
            b = lst[bi]
            
            # Early termination: if 1/(2b) >= best_error, smaller b in this fiber won't help
            if best_error > 0 and (1.0 / (2.0 * b)) >= best_error:
                break
            
            # Target a ≈ 1.5 * b
            target_a = 1.5 * b
            idx = bisect.bisect_left(lst, target_a)
            
            # Check a window around the target
            # Use a wider window to catch more candidates
            for delta in range(-30, 31):
                i = idx + delta
                if 0 <= i < len(lst):
                    a = lst[i]
                    if a == b:
                        continue
                    
                    num = abs(2 * a - 3 * b)
                    if num == 0:
                        continue
                    den = 2 * b
                    err = num / den
                    
                    if err < best_error:
                        best_error = err
                        best_a = a
                        best_b = b
    
    # Return as list of Python ints
    return [int(best_a), int(best_b)]
