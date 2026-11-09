import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # We need to find n in [10000, 1000000] that minimizes largest_prime_factor(n * (n+1))
    # Note: Since gcd(n, n+1) = 1, the prime factors of n(n+1) are the union of prime factors of n and n+1.
    # Thus, LPF(n(n+1)) = max(LPF(n), LPF(n+1)).
    
    # Strategy:
    # 1. Precompute smallest prime factor (SPF) for all numbers up to 1000001 using a sieve.
    # 2. Precompute largest prime factor for all numbers using SPF.
    # 3. Vectorized scan to find the n with minimum max(LPF(n), LPF(n+1)).
    
    MAX_N = 1000000
    LIMIT = MAX_N + 1  # We need to factor up to 1000001
    
    # Sieve for Smallest Prime Factor using NumPy
    spf = np.arange(LIMIT + 1, dtype=np.int32)
    spf[0] = 0
    spf[1] = 1
    
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if spf[i] == i:  # i is prime
            start = i * i
            step = i
            spf[start:LIMIT+1:step] = np.minimum(spf[start:LIMIT+1:step], i)
    
    # Precompute largest prime factor for all numbers
    lpf = np.ones(LIMIT + 1, dtype=np.int32)
    for n in range(2, LIMIT + 1):
        m = n
        while m > 1:
            p = spf[m]
            lpf[n] = p
            while m % p == 0:
                m //= p
    
    # Vectorized scan: find n that minimizes max(lpf[n], lpf[n+1])
    n_range = np.arange(10000, MAX_N + 1)
    lpf_n = lpf[n_range]
    lpf_n1 = lpf[n_range + 1]
    current_lpf = np.maximum(lpf_n, lpf_n1)
    
    best_idx = np.argmin(current_lpf)
    best_n = int(n_range[best_idx])
    best_lpf = int(current_lpf[best_idx])
    
    return best_n
# EVOLVE-BLOCK-END
