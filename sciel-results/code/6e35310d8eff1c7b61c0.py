"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve(n):
    """Return list of primes up to n."""
    if n < 2:
        return []
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, isqrt(n) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, n + 1) if is_prime[i]]

# Precompute primes up to 200 (max gap is 16, so we need to check divisibility by primes < 16)
# Actually, for a witness m with least prime factor >= gap, we need to verify that
# m is not divisible by any prime < gap.
# Max gap in our data: let's check
# Gaps: 2,28,2,22,6,2,10,12,2,6,4,8,22,6,2,10,8,4,6,8,4,12,18,12,20,4,6,6,8,4,2,16
# Max gap = 28

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin: 85s out of 90s limit
    
    n_gaps = len(P['gaps'])
    result = [None] * n_gaps
    
    # Precompute primes up to 30 (max gap is 28, so we need primes < 28)
    primes = _sieve(30)
    
    # For each gap, find a witness m strictly between lo and hi
    # such that least_prime_factor(m) >= gap
    # This means m is not divisible by any prime < gap
    
    for i in range(n_gaps):
        lo, hi = P['gaps'][i]
        gap = hi - lo
        
        # We need m in (lo, hi) such that no prime p < gap divides m
        # The relevant primes are those < gap
        relevant_primes = [p for p in primes if p < gap]
        
        found = False
        # Try candidates in the range (lo, hi)
        # Heuristic: try middle first, then alternate outward? Or just sequential?
        # For small gaps (2, 4, 6, 8, 10, 12, 16, 18, 20, 22, 28), there are
        # only 1-2 candidates in the gap. For gap=2, there's 1 candidate.
        # For larger gaps, there are more candidates.
        
        # Let's just iterate through all candidates in order
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            
            # Check if m has least prime factor >= gap
            # i.e., m is not divisible by any prime < gap
            ok = True
            for p in relevant_primes:
                if m % p == 0:
                    ok = False
                    break
            
            if ok:
                # Also need to verify that the least prime factor is actually >= gap
                # Since we checked all primes < gap, and none divide m,
                # the least prime factor must be >= gap (or m is prime, in which case lf = m >= gap since m > lo >= 10007 >> gap)
                result[i] = m
                found = True
                break
        
        if not found:
            result[i] = None
    
    # Ensure all values are Python int (not numpy or bool)
    for i in range(n_gaps):
        if result[i] is not None:
            result[i] = int(result[i])
    
    return result

# EVOLVE-BLOCK-END
