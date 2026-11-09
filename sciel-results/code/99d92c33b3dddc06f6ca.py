"""Rewritten program: find witnesses in consecutive prime gaps."""
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_primes(limit):
    """Return list of all primes <= limit."""
    if limit < 2:
        return []
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]


def construct():
    start_time = time.time()
    deadline = start_time + 80.0

    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n

    # Max gap size determines which primes we need to test divisibility against
    max_gap = max(hi - lo for lo, hi in gaps)
    # We need primes strictly less than the gap size
    small_primes = _sieve_primes(max_gap - 1)

    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        gap_size = hi - lo
        # Only need to test primes < gap_size
        test_primes = [p for p in small_primes if p < gap_size]
        found = None
        for m in range(lo + 1, hi):
            ok = True
            for p in test_primes:
                if m % p == 0:
                    ok = False
                    break
            if ok:
                found = m
                break
        if found is not None:
            result[i] = int(found)

    return result
