"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve(n):
    """Return a list of primes up to n."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

# Precompute primes up to 100 (max gap in P is 28, so primes < 28 are needed, but 100 is safe)
_PRIMES = _sieve(100)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        # Primes strictly less than gap
        small_primes = [p for p in _PRIMES if p < gap]
        
        found = False
        # Iterate through candidates in the gap
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
                
            # Check if m is divisible by any prime < gap
            is_witness = True
            for p in small_primes:
                if m % p == 0:
                    is_witness = False
                    break
            
            if is_witness:
                result[i] = m
                found = True
                break
                
    # Ensure types are correct (int or None)
    # The loop assigns int m or leaves None
    return result
# EVOLVE-BLOCK-END
