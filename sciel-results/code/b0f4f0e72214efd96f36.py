"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from math import isqrt

def construct():
    """
    Find integer n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy:
    1. Compute the largest prime factor (LPF) for every integer up to 1,000,001 using a sieve-like approach.
    2. For each n in [10000, 1000000], compute LPF(n * (n+1)) = max(LPF(n), LPF(n+1)).
    3. Find the n that minimizes this value.
    
    We use an array `lpf` where lpf[i] is the largest prime factor of i.
    We compute this efficiently using a segmented sieve approach for speed.
    """
    MAX_N = 1000000
    MAX_VAL = MAX_N + 1  # We need LPF up to 1000001
    
    # Initialize LPF array
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)
    lpf[1] = 1
    
    # Sieve for largest prime factor
    # For each number i from 2 to MAX_VAL, if it hasn't been assigned a prime factor yet (i.e., it's prime),
    # then for all multiples of i, set their largest prime factor to i.
    # Since we iterate i in increasing order, the last assignment to lpf[m] will be from the largest prime factor of m.
    
    # We only need to check up to sqrt(MAX_VAL) for primality, but we still need to mark all primes
    # Actually, we need to mark all primes up to MAX_VAL to correctly assign LPF for all numbers.
    # But we can optimize: for each prime p, we mark multiples. The key insight is that we only
    # need to iterate p up to MAX_VAL, but we can skip composites.
    
    # Efficient approach: use a simple sieve to find primes, then mark LPF
    # First, let's find all primes up to MAX_VAL using a standard sieve
    is_prime = np.ones(MAX_VAL + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, isqrt(MAX_VAL) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    # Now for each prime, mark its multiples' LPF
    for p in range(2, MAX_VAL + 1):
        if is_prime[p]:
            # Mark all multiples of p with p as their LPF
            # Since we go in increasing order, the last prime that marks a number is its largest prime factor
            lpf[p::p] = p
    
    # Now find the n in [10000, 1000000] that minimizes max(lpf[n], lpf[n+1])
    # We can vectorize this search
    n_values = np.arange(10000, MAX_N + 1)
    lpf_n = lpf[n_values]
    lpf_n1 = lpf[n_values + 1]
    
    # The LPF of n*(n+1) is max(LPF(n), LPF(n+1))
    combined_lpf = np.maximum(lpf_n, lpf_n1)
    
    # Find the index of the minimum value
    min_idx = np.argmin(combined_lpf)
    best_n = n_values[min_idx]
    best_val = combined_lpf[min_idx]
    
    return int(best_n)
# EVOLVE-BLOCK-END
