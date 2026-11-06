import time
import numpy as np

def construct():
    deadline = time.time() + 80
    target = 123457
    
    # 1. Compute Largest Prime Factor (LPF) using NumPy
    max_n = target - 1
    lpf = np.zeros(max_n + 1, dtype=np.int32)
    lpf[1] = 1
    
    # Sieve for largest prime factor
    # We iterate through primes and update multiples
    # This is O(N log log N)
    for i in range(2, max_n + 1):
        if lpf[i] == 0: # i is prime
            # Update all multiples of i
            # j = i, 2i, 3i, ...
            indices = np.arange(i, max_n + 1, i)
            lpf[indices] = i
            
        if time.time() > deadline:
            break

    # 2. Find the pair (a, b) with a+b=target that minimizes max(P(a), P(b))
    # We only need to check a up to target//2
    # Vectorize the check?
    # a ranges from 1 to target//2
    # b = target - a
    
    half = target // 2
    a_vals = np.arange(1, half + 1)
    b_vals = target - a_vals
    
    # Get LPF for these values
    # Note: lpf is 1-indexed for values 1..max_n
    # a_vals are 1..half
    # b_vals are target-1 .. target-half
    # All are within 1..max_n
    
    pa = lpf[a_vals]
    pb = lpf[b_vals]
    
    cur_maxes = np.maximum(pa, pb)
    
    # Find the index of the minimum value
    min_idx = np.argmin(cur_maxes)
    best_max = cur_maxes[min_idx]
    best_a = a_vals[min_idx]
    
    return [int(best_a), int(target - best_a)]
