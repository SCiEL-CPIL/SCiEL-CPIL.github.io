"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _primes_below(n):
    """Return list of primes less than n."""
    if n <= 2:
        return []
    sieve = bytearray(b'\x01') * n
    sieve[0] = 0
    sieve[1] = 0
    for i in range(2, isqrt(n - 1) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(n) if sieve[i]]

def _primorial(primes):
    """Compute product of given primes."""
    result = 1
    for p in primes:
        result *= p
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s to be safe under 90s limit
    
    result = [None] * len(P['gaps'])
    
    # Precompute primes up to max gap
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    small_primes = _primes_below(max_gap)
    
    # For each gap, compute the primorial of primes < gap
    # A number m is valid if gcd(m, primorial) == 1
    # This means m has no prime factor < gap, so least prime factor >= gap
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
        gap = hi - lo
        # Get primes strictly less than gap
        relevant_primes = [p for p in small_primes if p < gap]
        if not relevant_primes:
            # No small primes to exclude, any number works
            m = lo + 1
            result[i] = m
            continue
        
        prod = 1
        for p in relevant_primes:
            prod *= p
        
        # Find first m in (lo, hi) such that gcd(m, prod) == 1
        found = False
        for m in range(lo + 1, hi):
            # Check if m is coprime to prod
            # Using Euclidean algorithm
            a, b = m, prod
            while b:
                a, b = b, a % b
            if a == 1:
                result[i] = m
                found = True
                break
        
        if not found:
            result[i] = None
    
    return result
# EVOLVE-BLOCK-END
