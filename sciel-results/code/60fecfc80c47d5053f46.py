"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s to be safe under 90s kill
    target = 123457
    max_n = target - 1  # 123456
    
    # Vectorized sieve to compute largest prime factor
    lpf = np.zeros(max_n + 1, dtype=np.int32)
    lpf[1] = 1
    
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            # Mark all multiples of i
            indices = np.arange(i, max_n + 1, i)
            current = lpf[indices]
            mask = (current == 0) | (i > current)
            lpf[indices[mask]] = i
    
    # Vectorized search: compute max(lpf[a], lpf[target-a]) for all a in [1, target-1]
    a_arr = np.arange(1, target)
    b_arr = target - a_arr
    pa = lpf[a_arr]
    pb = lpf[b_arr]
    cur_max = np.maximum(pa, pb)
    
    best_idx = np.argmin(cur_max)
    best_max = int(cur_max[best_idx])
    best_a = int(a_arr[best_idx])
    best_b = int(b_arr[best_idx])
    
    # Try to improve by searching for pairs where both have smaller largest prime factors
    # We want to find a, b with a+b=target and max(P(a),P(b)) < best_max
    # This means both a and b must have largest prime factor < best_max
    # Let's try smaller thresholds
    
    for threshold in range(best_max - 1, 1, -1):
        if time.time() > deadline - 5:
            break
        
        # Find all a in [1, target-1] such that lpf[a] <= threshold and lpf[target-a] <= threshold
        # We can use the precomputed lpf array
        valid_a = (lpf[a_arr] <= threshold) & (lpf[b_arr] <= threshold)
        
        if np.any(valid_a):
            # Found a pair with max(P(a), P(b)) <= threshold
            # Pick the first one
            idx = np.argmax(valid_a)
            new_a = int(a_arr[idx])
            new_b = int(b_arr[idx])
            new_max = int(cur_max[idx])
            
            if new_max < best_max:
                best_a = new_a
                best_b = new_b
                best_max = new_max
            
            # Since we're going from high to low threshold, the first valid one at this threshold
            # gives us the best possible for this threshold level
            # But we should check if there's an even better one (lower max) at this threshold
            # Actually, since we're iterating threshold downward, the first threshold that works
            # gives us max(P(a),P(b)) <= threshold, but the actual max might be lower
            # Let's just take the minimum cur_max among valid indices
            valid_indices = np.where(valid_a)[0]
            if len(valid_indices) > 0:
                min_val = np.min(cur_max[valid_indices])
                min_idx = valid_indices[np.argmin(cur_max[valid_indices])]
                best_a = int(a_arr[min_idx])
                best_b = int(b_arr[min_idx])
                best_max = int(min_val)
                break  # We found the best possible at this threshold level
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
