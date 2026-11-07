"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    # 1. Find max value to sieve
    max_val = max(hi for lo, hi in P['gaps'])
    
    # 2. Sieve of Eratosthenes to get primes up to max_val
    # We need primes to identify small factors.
    is_prime = [True] * (max_val + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, isqrt(max_val) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 3. For each gap, find a witness
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap_size = hi - lo
        
        # Check time
        if time.time() > deadline:
            break
            
        # We need an integer m in (lo, hi) such that least_prime_factor(m) >= gap_size.
        # This means m is not divisible by any prime p < gap_size.
        
        # Optimization: Pre-compute the set of small primes to check
        # If gap_size is small, we check few primes.
        # If gap_size is large (e.g., 20), we check primes up to 19.
        
        # Iterate through candidates
        found = False
        for m in range(lo + 1, hi):
            # If m is prime, its least factor is itself (which is > gap_size usually, 
            # but actually we just need LPF >= gap_size).
            # If m is composite, we check its factors.
            
            # Quick check: if m < gap_size * 2, it might be prime? 
            # No, just check divisibility by primes < gap_size.
            
            # If m is not divisible by any prime < gap_size, then its least prime factor 
            # is either a prime >= gap_size or m itself (if prime).
            # In both cases, LPF(m) >= gap_size.
            
            valid = True
            for p in primes:
                if p >= gap_size:
                    break
                if m % p == 0:
                    valid = False
                    break
            
            if valid:
                result[i] = m
                found = True
                break
                
    return result

# EVOLVE-BLOCK-END
