"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _primes_up_to(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i, v in enumerate(sieve) if v]

# Precompute small primes up to 101 (sqrt(10300) ~ 101.5)
_SMALL_PRIMES = _primes_up_to(101)

def _has_small_prime_factor(m, threshold):
    """Check if m has any prime factor < threshold. Returns True if it does."""
    for p in _SMALL_PRIMES:
        if p >= threshold:
            break
        if m % p == 0:
            return True
    return False

def construct():
    start_time = time.time()
    DEADLINE = 85  # seconds, safety margin from 90s limit
    
    n = len(P['gaps'])
    result = [None] * n
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() - start_time > DEADLINE:
            break
        gap = hi - lo
        # Find all valid witnesses: m with no prime factor < gap
        # Try candidates in order, but also try from both ends for coverage
        found = False
        # Forward scan
        for m in range(lo + 1, hi):
            if not _has_small_prime_factor(m, gap):
                result[i] = m
                found = True
                break
        if not found:
            # Try reverse scan as well (shouldn't be needed but safety)
            for m in range(hi - 1, lo, -1):
                if not _has_small_prime_factor(m, gap):
                    result[i] = m
                    found = True
                    break
    
    # Ensure all are Python int or None
    return [int(x) if x is not None else None for x in result]
# EVOLVE-BLOCK-END
