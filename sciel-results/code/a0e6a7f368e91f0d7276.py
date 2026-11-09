"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _linear_sieve_lpf(limit):
    """Linear sieve to compute least prime factor for all integers up to limit."""
    lpf = list(range(limit + 1))
    lpf[0] = 0
    if limit >= 1:
        lpf[1] = 1
    primes = []
    for i in range(2, limit + 1):
        if lpf[i] == i:
            primes.append(i)
        for p in primes:
            if p > lpf[i] or i * p > limit:
                break
            lpf[i * p] = p
    return lpf

def construct():
    deadline = time.time() + 85.0  # 5s safety margin before 90s kill
    start_time = time.time()
    
    gaps = P['gaps']
    n = len(gaps)
    result = [None] * n
    
    # Precompute least prime factor for all integers up to max prime in gaps.
    max_val = max(hi for _, hi in gaps)
    lpf = _linear_sieve_lpf(max_val)
    
    # For each gap, find integers m in (lo, hi) such that
    # least_prime_factor(m) >= (hi - lo)
    # Use O(1) LPF lookup from the precomputed array.
    
    for i, (lo, hi) in enumerate(gaps):
        gap_size = hi - lo
        if time.time() > deadline:
            break
        # Try to find at least one valid witness m
        for m in range(lo + 1, hi):
            if lpf[m] >= gap_size:
                result[i] = m
                break
        # If no m found in this gap, result[i] remains None
    
    # Ensure types are correct (Python int, not numpy)
    for i in range(n):
        if result[i] is not None:
            result[i] = int(result[i])
    
    return result

# EVOLVE-BLOCK-END
