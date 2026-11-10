"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    MAX_N = 1000000
    MIN_N = 10000
    
    # Build smallest prime factor sieve up to MAX_N + 1
    spf = list(range(MAX_N + 2))
    for i in range(2, int(math.isqrt(MAX_N + 1)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_N + 2, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Compute powerful part for each number
    # powerful_part(n) = product of p^e where e >= 2 in factorization
    pp = [1] * (MAX_N + 2)
    spf_local = spf
    deadline_local = deadline
    
    for n in range(MIN_N, MAX_N + 1):
        x = n
        val = 1
        while x > 1:
            p = spf_local[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                val *= (p ** e)
        pp[n] = val
        if time.time() > deadline_local:
            break
    
    # Find best n using NumPy for vectorized max product
    # pp[MIN_N : MAX_N+1] contains powerful parts for n in [MIN_N, MAX_N]
    pp_arr = np.array(pp[MIN_N : MAX_N + 1], dtype=np.int64)
    
    if len(pp_arr) < 2:
        return MIN_N

    # Calculate products of adjacent powerful parts
    # products[i] = pp_arr[i] * pp_arr[i+1] corresponds to n = MIN_N + i
    products = pp_arr[:-1] * pp_arr[1:]
    
    # Find the index of the maximum product
    best_idx = np.argmax(products)
    best_n = MIN_N + best_idx
    
    return int(best_n)
# EVOLVE-BLOCK-END
