"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s to be safe under 90s kill
    target = 123457
    max_n = target - 1  # 123456
    
    # Sieve to compute largest prime factor
    lpf = np.zeros(max_n + 1, dtype=np.int32)
    lpf[1] = 1
    
    # Only need to iterate primes up to max_n, but for efficiency
    # we can stop early once i*i > max_n for the "prime" check,
    # but we still need to mark multiples for all primes.
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            # Mark all multiples of i that haven't been marked by a larger prime
            # Use step-based assignment
            for j in range(i, max_n + 1, i):
                if lpf[j] == 0 or i > lpf[j]:
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
    
    elapsed = time.time() - start_time
    if elapsed > 80:
        # Safety: return what we have
        return [best_a, best_b]
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
