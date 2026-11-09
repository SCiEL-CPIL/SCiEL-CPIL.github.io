# EVOLVE-BLOCK-START
import math
from math import isqrt
import time

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


def _has_small_factor(m, primes):
    """Check if m has any prime factor in the given list. Return True if it does."""
    for p in primes:
        if m % p == 0:
            return True
    return False


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s safety margin
    
    gaps = P['gaps']
    n_gaps = len(gaps)
    result = [None] * n_gaps
    
    max_hi = max(hi for _, hi in gaps)
    # Precompute primes up to max gap size for LPF checks
    max_gap = max(hi - lo for lo, hi in gaps)
    prime_list = _small_primes(max_gap)
    
    # For each gap, we need primes less than the gap size
    # A witness m must not be divisible by any prime < gap_size
    # So we check: is m divisible by any prime in prime_list where prime < gap_size?
    
    for idx, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
            
        gap_size = hi - lo
        # Get primes less than gap_size
        small_primes_for_gap = [p for p in prime_list if p < gap_size]
        
        # Search from the middle outward
        mid = (lo + hi) // 2
        candidates = []
        left = mid
        right = mid + 1
        count = 0
        max_candidates = gap_size - 1
        while count < max_candidates:
            if left > lo:
                candidates.append(left)
                left -= 1
                count += 1
            if right < hi:
                candidates.append(right)
                right += 1
                count += 1
            if left <= lo and right >= hi:
                break
        
        for m in candidates:
            if time.time() > deadline:
                break
            # Check if m has any prime factor < gap_size
            if not _has_small_factor(m, small_primes_for_gap):
                result[idx] = int(m)
                break
    
    return result


# EVOLVE-BLOCK-END
