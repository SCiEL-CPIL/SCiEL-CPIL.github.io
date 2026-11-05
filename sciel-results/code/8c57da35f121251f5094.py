"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find integer n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy:
    1. Compute the largest prime factor (LPF) for every integer up to 1,000,001 using a sieve-like approach.
    2. For each n in [10000, 1000000], compute LPF(n * (n+1)) = max(LPF(n), LPF(n+1)).
    3. Find the n that minimizes this value.
    
    We use an array `lpf` where lpf[i] is the largest prime factor of i.
    We can compute this efficiently by iterating through primes and updating multiples.
    """
    MAX_N = 1000000
    MAX_VAL = MAX_N + 1  # We need LPF up to 1000001
    
    # Initialize LPF array
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)
    lpf[0] = 0
    lpf[1] = 1
    
    # Sieve for largest prime factor using vectorized numpy operations
    # For each prime p, mark all multiples of p with p as their largest prime factor
    # Since we iterate in increasing order, the last assignment is the largest prime factor
    
    for p in range(2, MAX_VAL + 1):
        if lpf[p] == 0:  # p is prime
            # Vectorized update: mark all multiples of p
            multiples = np.arange(p, MAX_VAL + 1, p)
            lpf[multiples] = p
    
    # Now find the n in [10000, 1000000] that minimizes max(lpf[n], lpf[n+1])
    # Use vectorized operations for the search
    n_range = np.arange(10000, MAX_N + 1)
    vals = np.maximum(lpf[n_range], lpf[n_range + 1])
    best_idx = np.argmin(vals)
    best_n = n_range[best_idx]
    
    return int(best_n)
# EVOLVE-BLOCK-END
