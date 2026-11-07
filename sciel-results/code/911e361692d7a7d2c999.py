# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _primes_upto(n):
    """Return list of primes <= n (Sieve of Eratosthenes)."""
    if n < 2:
        return []
    sieve = bytearray([1]) * (n + 1)
    sieve[0] = sieve[1] = 0
    i = 2
    while i * i <= n:
        if sieve[i]:
            step = i
            start = i * i
            sieve[start:n + 1:step] = bytearray(len(range(start, n + 1, step)))
        i += 1
    return [i for i in range(2, n + 1) if sieve[i]]


def _least_prime_factor(m, primes):
    """Return least prime factor of m, or m if m is prime.
    primes must contain all primes up to sqrt(m)."""
    for p in primes:
        if p * p > m:
            break
        if m % p == 0:
            return p
    return m


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin

    gaps = P['gaps']
    result = [None] * len(gaps)

    # Precompute all primes up to max gap - 1 (for "bad" divisors)
    # and up to sqrt(max_candidate) (for LPF verification)
    max_gap = max(hi - lo for lo, hi in gaps)
    max_candidate = max(hi for lo, hi in gaps)
    sqrt_max = isqrt(max_candidate) + 1

    all_primes_small = _primes_upto(max_gap - 1)
    all_primes_verify = _primes_upto(sqrt_max)

    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break

        gap = hi - lo
        num_candidates = hi - lo - 1
        if num_candidates <= 0:
            continue

        # Get primes strictly less than gap (the "bad" divisors)
        small = [p for p in all_primes_small if p < gap]

        # Sieve-based approach: mark candidates NOT divisible by any small prime
        # clean[j] = 1 means candidate (lo+1+j) is not divisible by any p in small
        clean = bytearray([1]) * num_candidates

        for p in small:
            # First multiple of p in [lo+1, hi-1]
            first = ((lo + 1 + p - 1) // p) * p
            if first > hi - 1:
                continue
            start_idx = first - (lo + 1)
            # Mark all multiples
            for j in range(start_idx, num_candidates, p):
                clean[j] = 0

        # Find the first clean candidate
        found = None
        for j in range(num_candidates):
            if clean[j]:
                m = lo + 1 + j
                # Verify: compute LPF and check it's >= gap
                lpf = _least_prime_factor(m, all_primes_verify)
                if lpf >= gap:
                    found = m
                    break

        if found is not None:
            result[i] = int(found)

    # Ensure all entries are Python int or None
    final_result = []
    for entry in result:
        if entry is None:
            final_result.append(None)
        else:
            final_result.append(int(entry))

    return final_result
# EVOLVE-BLOCK-END
