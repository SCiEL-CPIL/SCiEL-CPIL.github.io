"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    target = 123457
    max_n = target - 1  # 123456
    
    # Compute largest prime factor using a sieve
    # For each prime p, mark all multiples with p as their largest prime factor
    # We want the largest prime factor, so we iterate primes in increasing order
    # and overwrite smaller ones with larger ones
    
    lpf = np.zeros(max_n + 1, dtype=np.int32)
    lpf[1] = 1
    
    # Use a more efficient approach: for each number, find its largest prime factor
    # by iterating through primes in order and marking multiples
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            # Mark all multiples of i with i as a prime factor
            # Since we go in increasing order, later (larger) primes will overwrite
            # So we just set lpf[j] = i for all multiples j of i
            for j in range(i, max_n + 1, i):
                lpf[j] = i
    
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
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
