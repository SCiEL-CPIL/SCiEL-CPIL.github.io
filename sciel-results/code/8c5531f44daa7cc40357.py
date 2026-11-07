# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _gen_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(limit + 1) if sieve[i]]

def _build_is_prime_array(lo, hi):
    """Return a bytearray where arr[k] is 1 iff lo+k is prime, for k in 0..hi-lo."""
    size = hi - lo + 1
    arr = bytearray(size)
    # Sieve over the range [lo, hi]
    # For each prime p, mark multiples in range
    small_primes = _gen_primes(isqrt(hi) + 1)
    for p in small_primes:
        # Find first multiple of p >= lo
        start = ((lo + p - 1) // p) * p
        if start < lo:
            start += p
        for m in range(start, hi + 1, p):
            arr[m - lo] = 0
    # Set primes
    for k in range(size):
        val = lo + k
        if val < 2:
            arr[k] = 0
        elif arr[k]:
            # Check if actually prime (not marked by sieve)
            # arr[k] is still 1 if not marked, meaning no prime <= sqrt(hi) divides it
            # Since sqrt(hi) < hi for hi > 1, if no factor found, it's prime
            pass  # already 1
        # If val >= 2 and arr[k] is 1, it's prime
    # Fix: numbers that are 0 because they're < 2 or marked
    # Actually the sieve marks composites. Let me redo more carefully.
    # Reset and do proper sieve
    arr = bytearray([1]) * size
    for k in range(size):
        val = lo + k
        if val < 2:
            arr[k] = 0
    # Sieve: for each prime p <= sqrt(hi), mark composites
    for p in small_primes:
        if p * p > hi:
            break
        # First multiple of p in [lo, hi]
        start = ((lo + p - 1) // p) * p
        if start < p * p:
            start = p * p
        for m in range(start, hi + 1, p):
            arr[m - lo] = 0
    return arr, small_primes

def _least_prime_factor(m, small_primes):
    """Return least prime factor of m, or m if m is prime."""
    for p in small_primes:
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

    # Precompute primality over the full range
    min_val = min(lo for lo, hi in gaps)
    max_val = max(hi for lo, hi in gaps)
    is_prime_arr, small_primes = _build_is_prime_array(min_val, max_val)

    for i, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
        gap_size = hi - lo
        found = False

        # First, check if any interior number is prime (lpf = itself >= gap_size always)
        for m in range(lo + 1, hi):
            if is_prime_arr[m - min_val]:
                result[i] = int(m)
                found = True
                break

        if found:
            continue

        # No prime in the gap; search for composite with lpf >= gap_size
        # Try candidates not divisible by any prime < gap_size
        primes_below_gap = [p for p in small_primes if p < gap_size]

        # Score candidates by how many small primes divide them (fewer is better)
        candidates = list(range(lo + 1, hi))
        scored = []
        for m in candidates:
            div_count = 0
            for p in primes_below_gap:
                if m % p == 0:
                    div_count += 1
            scored.append((div_count, m))

        scored.sort(key=lambda x: (x[0], x[1]))

        for _, m in scored:
            lpf = _least_prime_factor(m, small_primes)
            if lpf >= gap_size:
                result[i] = int(m)
                found = True
                break

        if not found:
            # Final fallback: linear scan
            for m in candidates:
                lpf = _least_prime_factor(m, small_primes)
                if lpf >= gap_size:
                    result[i] = int(m)
                    break

    # Verify all results
    for i, m in enumerate(result):
        if m is not None:
            lo, hi = gaps[i]
            gap_size = hi - lo
            if not (lo < m < hi):
                result[i] = None
            else:
                lpf = _least_prime_factor(m, small_primes)
                if lpf < gap_size:
                    result[i] = None
                else:
                    result[i] = int(m)

    return result
# EVOLVE-BLOCK-END
