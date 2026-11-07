"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve(n):
    """Return a list of primes up to n."""
    if n < 2:
        return []
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = 0
    return [i for i, prime in enumerate(is_prime) if prime]

# Precompute primes up to 100
_PRIMES = _sieve(100)

# Precompute for each distinct gap size the list of primes strictly less than that gap
_max_gap = max(hi - lo for lo, hi in P['gaps'])
_primes_by_gap = {}
for g in range(2, _max_gap + 1):
    _primes_by_gap[g] = [p for p in _PRIMES if p < g]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        small_primes = _primes_by_gap[gap]
        
        if not small_primes:
            # No primes < gap, so any m in (lo, hi) works
            result[i] = lo + 1
            continue
        
        # Try candidates in order; use vectorized check for speed
        candidates = np.arange(lo + 1, hi, dtype=np.int64)
        
        # Check if any candidate is valid (not divisible by any prime < gap)
        # For each small prime, create a mask of candidates NOT divisible by it
        # A candidate is valid if it's not divisible by ANY small prime
        valid_mask = np.ones(len(candidates), dtype=bool)
        for p in small_primes:
            valid_mask &= (candidates % p != 0)
        
        # Find first valid candidate
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) > 0:
            result[i] = int(candidates[valid_indices[0]])
    
    # Ensure types are correct (Python int or None)
    final_result = []
    for v in result:
        if v is None:
            final_result.append(None)
        else:
            final_result.append(int(v))
    
    return final_result
# EVOLVE-BLOCK-END
