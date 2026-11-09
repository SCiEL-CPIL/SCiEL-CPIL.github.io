"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    min_val = 1001
    max_val = 1000024
    size = max_val - min_val + 1
    
    # Vectorized sieve: for each prime, compute the p-part of each multiple
    factors = np.ones(size + 1, dtype=np.int64)  # 1-indexed
    numbers = np.arange(min_val, max_val + 1)  # 0-indexed: numbers[i] = min_val + i
    
    for p in _SMALL_PRIMES:
        # Find all multiples of p in range
        start = ((min_val + p - 1) // p) * p
        idxs = np.arange(start, max_val + 1, p) - min_val  # 0-indexed into numbers
        # For each multiple, compute the full power of p
        # We iterate by power level
        remaining = idxs.copy()
        power = 1
        while len(remaining) > 0:
            # Check which of these are divisible by p (they all should be for the first pass)
            # But after extracting, some won't be divisible by p anymore
            vals = numbers[remaining]
            mask = vals % p == 0
            if not np.any(mask):
                break
            new_remaining = remaining[mask]
            factors[new_remaining + 1] *= p  # 1-indexed
            remaining = new_remaining
            power += 1
            if power > 20:
                break
    
    # Convert to list for sliding window (or use NumPy array directly)
    # Sliding window to find minimum distinct count
    best_n = 1000
    best_val = float('inf')
    
    freq = defaultdict(int)
    distinct_count = 0
    
    for i in range(24):
        f = int(factors[i + 1])
        if freq[f] == 0:
            distinct_count += 1
        freq[f] += 1
    
    best_val = distinct_count
    best_n = 1000
    
    for s in range(1, 999001):
        out_f = int(factors[s])
        freq[out_f] -= 1
        if freq[out_f] == 0:
            distinct_count -= 1
            del freq[out_f]
        
        in_f = int(factors[s + 24])
        if freq[in_f] == 0:
            distinct_count += 1
        freq[in_f] += 1
        
        if distinct_count < best_val:
            best_val = distinct_count
            best_n = 1000 + s
    
    return best_n
# EVOLVE-BLOCK-END
