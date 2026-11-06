"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N_MIN = 10000
    N_MAX = 1000000
    limit = N_MAX + 1  # We need lpf up to N_MAX+1
    
    # 1. Sieve of Eratosthenes to find all primes up to limit
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = b'\x00' * ((limit - start) // step + 1)
    
    primes = np.array([i for i in range(2, limit + 1) if is_prime[i]], dtype=np.int32)
    
    # 2. Compute Largest Prime Factor (LPF) for each number up to limit using NumPy
    # lpf[x] will store the largest prime factor of x
    lpf = np.zeros(limit + 1, dtype=np.int32)
    for p in primes:
        # Vectorized assignment: lpf[p, 2p, 3p, ...] = p
        indices = np.arange(p, limit + 1, p)
        lpf[indices] = p
    
    # 3. For each n in [N_MIN, N_MAX], compute max(lpf[n], lpf[n+1])
    #    and find the n that minimizes this value.
    # Use NumPy for vectorized computation
    n_range = np.arange(N_MIN, N_MAX + 1)
    v1 = lpf[n_range]
    v2 = lpf[n_range + 1]
    combined = np.maximum(v1, v2)
    
    min_idx = np.argmin(combined)
    best_n = int(n_range[min_idx])
    best_val = int(combined[min_idx])
    
    return best_n
# EVOLVE-BLOCK-END
