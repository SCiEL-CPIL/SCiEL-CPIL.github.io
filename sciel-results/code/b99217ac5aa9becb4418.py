# EVOLVE-BLOCK-START
import math
from math import isqrt
import time
import numpy as np

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


def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s deadline, 10s safety margin
    
    gaps = P['gaps']
    n_gaps = len(gaps)
    result = [None] * n_gaps
    
    max_gap = max(hi - lo for lo, hi in gaps)
    prime_list = _small_primes(max_gap)
    
    for idx, (lo, hi) in enumerate(gaps):
        if time.time() > deadline:
            break
            
        gap_size = hi - lo
        # Get primes less than gap_size
        small_primes_for_gap = [p for p in prime_list if p < gap_size]
        
        # Range of candidates: lo+1 to hi-1
        n_candidates = gap_size - 1
        if n_candidates <= 0:
            continue
        
        # Create boolean array: True means valid (no small factor)
        valid = np.ones(n_candidates, dtype=bool)
        
        # For each small prime, mark its multiples as invalid
        for p in small_primes_for_gap:
            # Find first multiple of p in range [lo+1, hi-1]
            first_mult = ((lo + 1 + p - 1) // p) * p
            if first_mult >= hi:
                continue
            # Mark all multiples of p in range
            start_idx = first_mult - (lo + 1)
            valid[start_idx::p] = False
        
        # Find any valid candidate
        valid_indices = np.where(valid)[0]
        if len(valid_indices) > 0:
            # Pick the middle one for balance
            mid_idx = len(valid_indices) // 2
            m = lo + 1 + int(valid_indices[mid_idx])
            result[idx] = int(m)
    
    return result


# EVOLVE-BLOCK-END
