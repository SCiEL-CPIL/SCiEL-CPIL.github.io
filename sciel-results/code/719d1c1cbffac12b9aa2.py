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

def _least_prime_factor(m, primes):
    """Return least prime factor of m using given primes, or m if none found (m is prime or LPF > max(primes))."""
    for p in primes:
        if p * p > m:
            return m
        if m % p == 0:
            return p
    return m

def construct():
    deadline = time.time() + 80  # 80s safety margin within 90s limit
    result = [None] * len(P['gaps'])
    
    # Max gap among all pairs
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    
    # We need primes up to sqrt(max candidate) for LPF checks, but actually
    # we only need primes < g for each gap to check if m has a factor < g.
    # For LPF >= g, we need no prime < g divides m.
    # The largest g is max_gap, so we need primes < max_gap.
    # But also, if m is prime, LPF = m which is > 10000 >> g, so that's fine.
    # If m is composite, LPF could be anything. We need LPF >= g.
    # So we check: does any prime p < g divide m? If not, LPF >= g (or m is prime).
    
    # Precompute primes up to max_gap (so we have all primes < g for any gap)
    small_primes = _small_primes_up_to(max_gap)
    
    # For each gap, find a witness
    for i, (lo, hi) in enumerate(P['gaps']):
        g = hi - lo
        # We need m in (lo, hi) with no prime factor < g
        # Strategy: try candidates in the gap, check divisibility by primes < g
        
        # Collect primes < g
        primes_below_g = [p for p in small_primes if p < g]
        
        found = False
        # Try each candidate in the gap
        for m in range(lo + 1, hi):
            # Check if m has any prime factor < g
            has_small = False
            for p in primes_below_g:
                if m % p == 0:
                    has_small = True
                    break
            if not has_small:
                # m's least prime factor is >= g
                # Verify: if m is prime, LPF = m >= 10007 > g (since max g is 20)
                # If m is composite, its LPF is >= g by construction
                result[i] = int(m)
                found = True
                break
        
        if time.time() > deadline:
            break
    
    # Ensure all entries are Python int or None
    for i in range(len(result)):
        if result[i] is not None:
            result[i] = int(result[i])
    
    return result
# EVOLVE-BLOCK-END
