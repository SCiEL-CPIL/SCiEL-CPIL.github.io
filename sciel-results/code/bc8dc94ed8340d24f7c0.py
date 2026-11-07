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
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(limit + 1) if sieve[i]]

def _build_lpf_table(max_n, primes):
    """Build a table of least prime factors for all numbers up to max_n."""
    lpf = list(range(max_n + 1))
    for p in primes:
        if p * p > max_n:
            break
        for multiple in range(p * p, max_n + 1, p):
            if lpf[multiple] == multiple:
                lpf[multiple] = p
    return lpf

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Precompute primes up to sqrt(max value in gaps) - actually up to max_val for safety
    max_val = max(hi for _, hi in P['gaps'])
    small_primes = _gen_primes(isqrt(max_val) + 10)
    
    # Build LPF table for all numbers up to max_val
    lpf_table = _build_lpf_table(max_val, small_primes)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Find m in (lo, hi) such that LPF(m) >= gap
        # Try all candidates in the gap
        for m in range(lo + 1, hi):
            lpf = lpf_table[m]
            if lpf >= gap:
                result[i] = int(m)
                break
    
    # Convert to ensure Python int types
    final_result = []
    for val in result:
        if val is None:
            final_result.append(None)
        else:
            final_result.append(int(val))
    
    return final_result

# EVOLVE-BLOCK-END
