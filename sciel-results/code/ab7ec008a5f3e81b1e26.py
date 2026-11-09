"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_primes_up_to(n):
    """Return list of all primes <= n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]


def _least_prime_factor(m, primes):
    """Return the least prime factor of m, or m itself if prime."""
    if m < 2:
        return m
    limit = isqrt(m)
    for p in primes:
        if p > limit:
            break
        if m % p == 0:
            return p
    return m


def construct():
    deadline = time.monotonic() + 85.0  # safety margin before 90s kill
    result = [None] * len(P['gaps'])
    
    # Precompute primes up to sqrt of max possible value
    max_val = max(hi for _, hi in P['gaps'])
    primes = _sieve_primes_up_to(isqrt(max_val))
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.monotonic() > deadline:
            break
        gap = hi - lo
        # Search for a witness strictly between lo and hi whose least prime factor >= gap
        # We want at least one witness per gap; iterate candidates
        # Try candidates in a smart order: start from the middle, or try all
        found = False
        for m in range(lo + 1, hi):
            if time.monotonic() > deadline:
                break
            lpf = _least_prime_factor(m, primes)
            if lpf >= gap:
                result[i] = m
                found = True
                break
        if not found:
            # Try from the other end as well, or just leave as None
            # But first try was exhaustive, so None is correct
            pass
    
    # Ensure types are correct: integers must be Python int
    # The loop already assigns Python ints from range()
    return result
# EVOLVE-BLOCK-END
