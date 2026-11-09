"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _get_small_primes(limit):
    """Return list of primes < limit."""
    if limit <= 2:
        return []
    sieve = bytearray(b'\x01') * (limit)
    sieve[0] = 0
    sieve[1] = 0
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            sieve[i*i:limit:i] = bytearray(len(sieve[i*i:limit:i]))
    return [i for i, is_prime in enumerate(sieve) if is_prime]

def construct():
    # Determine max gap to precompute primes
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    # We need primes < g. So we need primes up to max_gap.
    # Actually, if g=28, we need primes < 28.
    small_primes = _get_small_primes(max_gap + 1)
    
    result = [None] * len(P['gaps'])
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85  # 5s margin for 90s limit
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        # Find witness m in (lo, hi) such that lpf(m) >= gap
        # This is equivalent to: no prime q < gap divides m
        
        # Optimization: Precompute relevant primes for this specific gap
        # Primes q such that q < gap
        relevant_primes = [p for p in small_primes if p < gap]
        
        found = False
        for m in range(lo + 1, hi):
            # Check if any relevant prime divides m
            has_small_factor = False
            for p in relevant_primes:
                if m % p == 0:
                    has_small_factor = True
                    break
            if not has_small_factor:
                result[i] = m
                found = True
                break
                
    return result
# EVOLVE-BLOCK-END
