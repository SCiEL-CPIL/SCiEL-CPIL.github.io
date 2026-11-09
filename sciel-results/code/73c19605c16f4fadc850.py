"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    deadline = time.time() + 80.0  # safety margin before 90s kill
    
    MAX_N = 1000000
    MAX_VAL = MAX_N + 1  # we need lpf up to 1000001
    
    # Sieve for largest prime factor
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)
    
    # Use a more efficient sieve: only iterate over primes
    # We'll use a standard sieve to find primes first, then compute lpf
    is_prime = np.ones(MAX_VAL + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(np.sqrt(MAX_VAL)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    # Now compute largest prime factor for each number
    # For each prime, update lpf for all its multiples
    primes = np.where(is_prime)[0]
    
    for p in primes:
        # For all multiples of p, set lpf to at least p
        # Since we go in increasing order of p, the last one that divides is the largest
        for j in range(p, MAX_VAL + 1, p):
            lpf[j] = p
    
    # Find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    # Vectorized scan
    vals = np.maximum(lpf[10000:MAX_N + 1], lpf[10001:MAX_N + 2])
    
    min_idx = np.argmin(vals)
    best_n = 10000 + min_idx
    best_val = vals[min_idx]
    
    return int(best_n)
# EVOLVE-BLOCK-END
