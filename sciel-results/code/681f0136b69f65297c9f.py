"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    from math import isqrt
    import time
    
    deadline = time.time() + 80  # safety margin before 90s limit
    
    # Precompute primes up to max sqrt of any number in the gaps
    # Numbers are around 10000-10289, so sqrt is at most ~102
    max_val = max(hi for lo, hi in P['gaps'])
    limit = isqrt(max_val) + 1
    
    # Sieve of Eratosthenes for primes up to limit
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(limit) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        gap = hi - lo
        found = False
        
        # Search for m in (lo, hi) whose least prime factor >= gap
        # If gap is large, m must be prime (since lp(m) >= gap > sqrt(m) for small gaps)
        # Actually, if gap > isqrt(m), then lp(m) >= gap implies m is prime
        # For smaller gaps, m could be composite with all factors >= gap
        
        for m in range(lo + 1, hi):
            if time.time() > deadline:
                break
            
            # Find least prime factor of m
            if m % 2 == 0:
                lp = 2
            else:
                lp = m  # default: m is prime
                for p in primes:
                    if p * p > m:
                        break
                    if m % p == 0:
                        lp = p
                        break
            
            if lp >= gap:
                result[i] = m
                found = True
                break
        
        if time.time() > deadline:
            break
    
    # Ensure all values are Python int, not numpy types
    final_result = []
    for v in result:
        if v is None:
            final_result.append(None)
        else:
            final_result.append(int(v))
    
    return final_result
# EVOLVE-BLOCK-END
