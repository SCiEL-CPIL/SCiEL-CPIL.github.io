"""Rewritten program for the suite task; stronger algorithm for finding witnesses."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve(limit):
    """Return bytearray where is_prime[i] is 1 iff i is prime, for 0<=i<=limit."""
    if limit < 2:
        return bytearray(limit + 1)
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i*i : limit+1 : i] = bytearray(len(range(i*i, limit+1, i)))
    return is_prime

def _small_primes_up_to(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    sp = _sieve(n)
    return [i for i in range(2, n + 1) if sp[i]]

def construct():
    deadline = time.time() + 80
    result = [None] * len(P['gaps'])
    
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    small_primes = _small_primes_up_to(max_gap)
    
    # Precompute primes below each possible gap size
    # Max gap is 28, so we can precompute for 2..28
    primes_below = {g: [p for p in small_primes if p < g] for g in range(2, max_gap + 1)}
    
    for i, (lo, hi) in enumerate(P['gaps']):
        g = hi - lo
        p_list = primes_below.get(g, [])
        
        # Try each candidate
        for m in range(lo + 1, hi):
            valid = True
            for p in p_list:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = int(m)
                break
        
        if time.time() > deadline:
            break
    
    return result
# EVOLVE-BLOCK-END
