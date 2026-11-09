import numpy as np
import time
import math
import bisect

def construct():
    t_start = time.time()
    deadline = t_start + 55.0  # Safety margin for 60s limit
    
    MAX_N = 10_000_000
    
    # 1. Compute sigma using optimized NumPy sieve
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Batch the additions for better cache performance
    # Process d in chunks where vector lengths are similar
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d
        
    if time.time() > deadline - 10:
        return [6, 11]

    # 2. Group by sigma value using a more memory-efficient approach
    # Convert to Python list for faster iteration
    sigma_list = sigma.tolist()
    
    buckets = {}
    for i in range(1, MAX_N + 1):
        s = sigma_list[i]
        if s in buckets:
            buckets[s].append(i)
        else:
            buckets[s] = [i]
            
    if time.time() > deadline - 8:
        return [6, 11]

    # 3. Find best pair with bidirectional search
    best_err = float('inf')
    best_pair = (6, 11)
    
    # Sort all bucket values for binary search
    for s in buckets:
        if len(buckets[s]) >= 2:
            buckets[s].sort()
    
    # For each bucket, search for pairs where a/b ≈ 3/2
    # This means a ≈ 1.5*b, so we need to check both directions:
    # - For each b, find a closest to 1.5*b
    # - For each a, find b closest to 2*a/3
    
    for s, vals in buckets.items():
        if len(vals) < 2:
            continue
            
        # Direction 1: for each b, find a ≈ 1.5*b
        for b in vals:
            if time.time() > deadline - 3:
                return [int(best_pair[0]), int(best_pair[1])]
                
            target_a = 3 * b // 2  # Integer division for speed
            # Also check nearby values to handle rounding
            for offset in [-1, 0, 1]:
                a_candidate = target_a + offset
                if a_candidate < 1 or a_candidate > MAX_N:
                    continue
                # Check if a_candidate is in vals using binary search
                idx = bisect.bisect_left(vals, a_candidate)
                if idx < len(vals) and vals[idx] == a_candidate:
                    a = vals[idx]
                    if a != b:
                        diff = abs(2 * a - 3 * b)
                        if diff > 0:
                            err = diff / (2.0 * b)
                            if err < best_err:
                                best_err = err
                                best_pair = (a, b)
                
            # Also do a proper binary search for the closest value
            target_float = 1.5 * b
            idx = bisect.bisect_left(vals, target_float)
            for idx_check in [idx - 1, idx]:
                if 0 <= idx_check < len(vals):
                    a = vals[idx_check]
                    if a != b:
                        diff = abs(2 * a - 3 * b)
                        if diff > 0:
                            err = diff / (2.0 * b)
                            if err < best_err:
                                best_err = err
                                best_pair = (a, b)
        
        # Direction 2: for each a, find b ≈ 2*a/3
        for a in vals:
            if time.time() > deadline - 3:
                return [int(best_pair[0]), int(best_pair[1])]
                
            target_b = 2 * a // 3  # Integer division for speed
            for offset in [-1, 0, 1]:
                b_candidate = target_b + offset
                if b_candidate < 1 or b_candidate > MAX_N:
                    continue
                idx = bisect.bisect_left(vals, b_candidate)
                if idx < len(vals) and vals[idx] == b_candidate:
                    b = vals[idx]
                    if a != b:
                        diff = abs(2 * a - 3 * b)
                        if diff > 0:
                            err = diff / (2.0 * b)
                            if err < best_err:
                                best_err = err
                                best_pair = (a, b)
                
            # Proper binary search
            target_float = 2.0 * a / 3.0
            idx = bisect.bisect_left(vals, target_float)
            for idx_check in [idx - 1, idx]:
                if 0 <= idx_check < len(vals):
                    b = vals[idx_check]
                    if a != b:
                        diff = abs(2 * a - 3 * b)
                        if diff > 0:
                            err = diff / (2.0 * b)
                            if err < best_err:
                                best_err = err
                                best_pair = (a, b)
    
    return [int(best_pair[0]), int(best_pair[1])]
