"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    N = 1000001  # we need lpf up to 1000001 (for n+1 when n=1000000)
    
    # Sieve of Eratosthenes using NumPy
    is_prime = np.ones(N + 1, dtype=np.bool_)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    # Build largest prime factor table using NumPy
    # lpf[m] = largest prime factor of m
    lpf = np.zeros(N + 1, dtype=np.int32)
    
    # For each prime p, set lpf[m] = p for all multiples m of p
    # Since we go in increasing order of p, the last assignment gives the largest prime factor
    primes = np.where(is_prime)[0]
    for p in primes:
        if p < 2:
            continue
        lpf[p::p] = p
    
    # Check time
    if time.time() > deadline - 5:
        # Not enough time, return a reasonable default
        return 10000
    
    # Find n in [10000, 1000000] that minimizes max(lpf[n], lpf[n+1])
    # Use vectorized NumPy operations
    n_range = np.arange(10000, 1000001)
    vals = np.maximum(lpf[n_range], lpf[n_range + 1])
    
    best_idx = np.argmin(vals)
    best_n = int(n_range[best_idx])
    best_val = int(vals[best_idx])
    
    return best_n
# EVOLVE-BLOCK-END
