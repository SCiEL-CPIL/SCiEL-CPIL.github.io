"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt
from functools import lru_cache

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

# Precompute small primes up to a reasonable limit for factor checking
_SMALL_PRIME_LIMIT = 200
def _gen_small_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

_SMALL_PRIMES = _gen_small_primes(_SMALL_PRIME_LIMIT)

def _least_prime_factor(m):
    """Return the least prime factor of m, or m if m is prime."""
    if m % 2 == 0:
        return 2
    for p in _SMALL_PRIMES:
        if p * p > m:
            return m
        if m % p == 0:
            return p
    return m

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    # For each gap, find m in (lo, hi) with lpf(m) >= gap_size
    # Key insight: m must not be divisible by any prime < gap_size
    # Generate candidates that are coprime to all primes < gap_size
    
    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        gap_size = hi - lo
        
        # Get primes that are < gap_size (these are forbidden divisors)
        forbidden_primes = [p for p in _SMALL_PRIMES if p < gap_size]
        
        # Generate candidates in (lo, hi) that are not divisible by any forbidden prime
        # Use Chinese Remainder Theorem style: try offsets from lo+1
        # Simpler: just iterate but skip numbers divisible by forbidden primes
        
        found = False
        # Try all m in range, but check efficiently
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            # Quick check: is m divisible by any forbidden prime?
            valid = True
            for p in forbidden_primes:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                # m is not divisible by any prime < gap_size
                # So lpf(m) >= gap_size (either m is prime or its smallest factor >= gap_size)
                result[i] = m
                found = True
                break
        
        if found:
            continue

    # Verify results
    for i, m in enumerate(result):
        if m is not None:
            lo, hi = gaps[i]
            gap_size = hi - lo
            if not (lo < m < hi):
                result[i] = None
            elif _least_prime_factor(m) < gap_size:
                result[i] = None
            elif not isinstance(m, int):
                result[i] = int(m)

    return result
# EVOLVE-BLOCK-END
