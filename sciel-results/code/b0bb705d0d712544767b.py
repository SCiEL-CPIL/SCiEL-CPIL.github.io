import time
import numpy as np
from math import isqrt

def construct():
    t0 = time.time()
    deadline = t0 + 55.0

    N_MIN = 10000
    N_MAX = 1000000
    upper = N_MAX + 1  # 1000001

    # Step 1: Build largest prime factor array via a segmented/standard sieve
    # This gives lpf[x] = largest prime factor of x for x in [2, upper]
    lpf = np.zeros(upper + 1, dtype=np.int64)
    
    # Efficient sieve for largest prime factor
    # For each prime p, mark multiples with p
    # We iterate primes in increasing order, so the last assignment wins
    # (which gives the largest prime factor)
    for i in range(2, upper + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, upper + 1, i):
                lpf[j] = i

    # Step 2: Find n in [N_MIN, N_MAX] minimizing max(lpf[n], lpf[n+1])
    n_range = np.arange(N_MIN, N_MAX + 1)
    scores = np.maximum(lpf[n_range], lpf[n_range + 1])
    
    best_idx = np.argmin(scores)
    best_n = int(n_range[best_idx])
    
    return best_n
