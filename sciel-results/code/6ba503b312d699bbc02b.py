"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]
_NUM_SMALL = len(_SMALL_PRIMES)

def _precompute_prime_masks(lo, hi):
    """For each integer in [lo, hi], compute a bitmask of which small primes divide it."""
    size = hi - lo + 1
    masks = np.zeros(size, dtype=np.int32)
    for idx, p in enumerate(_SMALL_PRIMES):
        bit = 1 << idx
        # Find all multiples of p in [lo, hi]
        first = ((lo + p - 1) // p) * p
        if first > hi:
            continue
        for m in range(first, hi + 1, p):
            masks[m - lo] |= bit
    return masks

def _count_distinct_primes(masks, lo):
    """Count distinct primes appearing in the 24 consecutive numbers starting at lo."""
    # masks[0:24] correspond to lo, lo+1, ..., lo+23
    union = 0
    for i in range(24):
        union |= masks[i]
    return bin(union).count('1')

def construct():
    deadline = time.time() + 80  # safety margin
    
    lo_bound = 1000
    hi_bound = 1000000
    
    # We need n+i for i=1..24, so we need numbers from n+1 to n+24
    # n ranges from 1000 to 1000000
    # So we need to check numbers from 1001 to 1000024
    
    # Precompute masks for the entire range we need
    # Numbers needed: 1001 to 1000024
    start = 1001
    end = 1000024
    size = end - start + 1
    
    masks = _precompute_prime_masks(start, end)
    
    best_n = 1000
    best_count = _NUM_SMALL + 1  # worse than max possible
    
    # Sliding window: for n, we look at masks[0:24] (numbers n+1 to n+24)
    # For n+1, we look at masks[1:25], etc.
    
    # Initial window for n=1000: numbers 1001..1024 -> masks[0:24]
    # Use a frequency count of each prime in the window
    prime_freq = [0] * _NUM_SMALL
    for i in range(24):
        m = masks[i]
        for idx in range(_NUM_SMALL):
            if m & (1 << idx):
                prime_freq[idx] += 1
    
    distinct = sum(1 for f in prime_freq if f > 0)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide the window
    # n goes from 1000 to 1000000
    # For n, window is masks[n+1-start : n+1-start+24] = masks[n-999 : n-999+24]
    # n=1000 -> masks[1:25]... wait let me re-index
    # start = 1001, so number x corresponds to masks[x - start] = masks[x - 1001]
    # For n, we need numbers n+1 to n+24
    # n+1 -> masks[n+1-1001] = masks[n-1000]
    # n+24 -> masks[n+24-1001] = masks[n-1000+23]
    # So window is masks[n-1000 : n-1000+24]
    
    # n=1000: masks[0:24]
    # n=1001: masks[1:25]
    # n=1000000: masks[999000:999024]
    
    max_n = 1000000
    for n in range(1001, max_n + 1):
        if time.time() > deadline:
            break
        
        # Remove masks[n-1000-1] (the one leaving)
        # Add masks[n-1000+23] (the one entering)
        leaving_idx = n - 1000 - 1  # = n - 1001
        entering_idx = n - 1000 + 23  # = n - 977
        
        # Wait, let me re-derive:
        # For n, window is masks[n-1000 : n-1000+24]
        # For n-1, window is masks[n-1001 : n-1001+24] = masks[n-1001 : n-977]
        # So going from n-1 to n:
        #   Remove index n-1001 (leaving)
        #   Add index n-977 (entering)
        
        leaving = masks[n - 1001]
        entering = masks[n - 977]
        
        for idx in range(_NUM_SMALL):
            bit = 1 << idx
            if leaving & bit:
                prime_freq[idx] -= 1
            if entering & bit:
                prime_freq[idx] += 1
        
        distinct = sum(1 for f in prime_freq if f > 0)
        if distinct < best_count:
            best_count = distinct
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
