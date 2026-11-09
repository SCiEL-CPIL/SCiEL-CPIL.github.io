# EVOLVE-BLOCK-START
import math
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _small_primes(limit):
    """Return list of primes up to limit using simple sieve."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for p in range(2, isqrt(limit) + 1):
        if sieve[p]:
            sieve[p*p::p] = bytearray(len(sieve[p*p::p]))
    return [i for i in range(limit + 1) if sieve[i]]


def _min_prime_factor_trial(n, primes):
    """
    Return the least prime factor of n using trial division by primes.
    If n is prime, return n itself.
    """
    for p in primes:
        if p * p > n:
            return n
        if n % p == 0:
            return p
    return n


def construct():
    gaps = P['gaps']
    n_gaps = len(gaps)
    result = [None] * n_gaps

    max_hi = max(hi for _, hi in gaps)
    # Need primes up to isqrt(max_hi) for trial division
    prime_list = _small_primes(isqrt(max_hi))

    for idx, (lo, hi) in enumerate(gaps):
        gap_size = hi - lo
        # Search from the middle outward to find a witness
        mid = (lo + hi) // 2
        # Build a sequence of candidates: mid, mid-1, mid+1, mid-2, mid+2, ...
        candidates = []
        left = mid
        right = mid + 1
        step = 0
        while len(candidates) < gap_size - 1:
            if left > lo:
                candidates.append(left)
                left -= 1
            if right < hi:
                candidates.append(right)
                right += 1
            step += 1
            if left <= lo and right >= hi:
                break

        for m in candidates:
            lpf = _min_prime_factor_trial(m, prime_list)
            if lpf >= gap_size:
                result[idx] = int(m)
                break

    return result


# EVOLVE-BLOCK-END
