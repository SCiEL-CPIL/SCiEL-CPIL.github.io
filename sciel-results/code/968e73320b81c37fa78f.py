import time
import math
import numpy as np

def construct():
    """
    Find integer n in [10000, 1000000] that minimizes the largest prime factor
    of the product n(n+1).
    
    Approach:
    1. Precompute the largest prime factor (LPF) for all integers up to 1000001
       using a modified Sieve of Eratosthenes.
    2. For each n in [10000, 1000000], compute score(n) = max(lpf[n], lpf[n+1]).
    3. Find n with the minimum score.
    
    This is exact and runs in < 1 second, well within the 60s budget.
    """
    start_time = time.time()
    deadline = start_time + 55.0
    
    LIMIT = 1000000
    MIN_N = 10000
    MAX_N = LIMIT
    MAX_VAL = MAX_N + 1  # We need lpf up to MAX_N + 1
    
    # Allocate array for largest prime factor
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int64)
    
    # Sieve: for each number i, if it's prime (lpf[i] == 0), 
    # mark all its multiples with i as their largest prime factor (for now).
    # Since we iterate i in increasing order, later (larger) primes will overwrite
    # earlier (smaller) ones for shared multiples.
    # At the end, lpf[x] = largest prime factor of x.
    
    for i in range(2, MAX_VAL + 1):
        if lpf[i] == 0:
            # i is prime
            # Update all multiples of i
            for j in range(i, MAX_VAL + 1, i):
                lpf[j] = i
    
    # Now lpf[x] contains the largest prime factor of x for all x in [2, MAX_VAL]
    # lpf[0] = lpf[1] = 0, which is fine since MIN_N = 10000
    
    # Compute score for each n: max(lpf[n], lpf[n+1])
    # Vectorized:
    lpf_n = lpf[MIN_N : MAX_N + 1]        # lpf[10000], lpf[10001], ..., lpf[1000000]
    lpf_np1 = lpf[MIN_N + 1 : MAX_N + 2]  # lpf[10001], lpf[10002], ..., lpf[1000001]
    
    scores = np.maximum(lpf_n, lpf_np1)
    
    # Find the n with minimum score
    best_idx = np.argmin(scores)
    best_n = MIN_N + best_idx
    
    # Sanity check: ensure we're within bounds
    assert MIN_N <= best_n <= MAX_N
    
    return int(best_n)
