"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, isqrt(n) + 1):
        if is_p[i]:
            for j in range(i*i, n+1, i):
                is_p[j] = False
    return [i for i, v in enumerate(is_p) if v]

def _has_small_factor(m, primes):
    """Check if m has any prime factor from the given list of primes."""
    for p in primes:
        if p * p > m:
            break
        if m % p == 0:
            return True
    return False

def construct():
    deadline = time.time() + 50  # safety margin before 60s limit
    
    # Max gap in our data is 20, so primes up to 20 suffice
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    primes = _sieve(max_gap)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # We need m with least prime factor >= gap
        # So m must not be divisible by any prime < gap
        small_primes = [p for p in primes if p < gap]
        
        for m in range(lo + 1, hi):
            if not _has_small_factor(m, small_primes):
                result[i] = int(m)
                break
    
    return result
# EVOLVE-BLOCK-END
