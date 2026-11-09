"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}


def _sieve_primes_up_to(n):
    """Return list of all primes <= n."""
    if n < 2:
        return []
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start::step] = bytearray(len(sieve[start::step]))
    return [i for i, is_prime in enumerate(sieve) if is_prime]


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Precompute primes up to a reasonable bound
    # The maximum number we'll check is around 10289, and we need to find
    # least prime factors. For numbers up to ~10289, we need primes up to sqrt(10289) ~ 101
    # But actually, we need to check if the least prime factor is >= gap size.
    # The largest gap here is 10289-10273=16. So we need least prime factor >= 16,
    # meaning the number is not divisible by 2,3,5,7,11,13.
    # So we need primes up to at least 13 to check divisibility.
    # But to be safe and general, let's compute primes up to max gap or sqrt(max_value).
    
    max_val = max(hi for _, hi in P['gaps'])
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    
    # We need to test divisibility by all primes less than the gap size for each gap.
    # Actually, we need the least prime factor to be >= gap.
    # So for a number m in (lo, hi) with gap g = hi - lo, we need lp(m) >= g.
    # This means m is not divisible by any prime < g.
    # So we need primes up to at least max_gap - 1 to check.
    
    primes_needed = _sieve_primes_up_to(max(max_gap, isqrt(max_val)) + 1)
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        # Find primes that are < gap (these are the ones that would disqualify m)
        small_primes = [p for p in primes_needed if p < gap]
        
        # Search for a witness: an integer m strictly between lo and hi
        # such that m is not divisible by any prime in small_primes
        # (i.e., least prime factor of m >= gap)
        
        found = False
        for m in range(lo + 1, hi):
            # Check if m has any prime factor < gap
            valid = True
            for p in small_primes:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                result[i] = m
                found = True
                break
        
        # Check time budget
        if time.time() > deadline - 1.0:
            break
    
    # Ensure all entries are Python int or None
    for i in range(len(result)):
        if result[i] is not None:
            result[i] = int(result[i])
    
    return result
# EVOLVE-BLOCK-END
