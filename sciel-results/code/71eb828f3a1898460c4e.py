# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _gen_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray([1]) * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    
    # Precompute primes up to sqrt(max possible m) ~ sqrt(10289) ~ 102
    max_m = max(hi for _, hi in P['gaps'])
    primes = _gen_primes(isqrt(max_m) + 1)
    
    def _least_prime_factor(m):
        """Return least prime factor of m, or m if prime."""
        for p in primes:
            if p * p > m:
                return m  # m is prime
            if m % p == 0:
                return p
        return m
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Search ascending from lo+1
        found = False
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            lp = _least_prime_factor(m)
            if lp >= gap:
                result[i] = int(m)
                found = True
                break
        if not found:
            # If not found ascending (shouldn't happen if we checked all), 
            # the ascending loop already covered all values, so leave as None
            pass
    
    return result
# EVOLVE-BLOCK-END
