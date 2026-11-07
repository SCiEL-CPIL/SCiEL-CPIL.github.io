import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def _sieve(n):
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    # Max gap is 28, so we need primes up to 27
    primes = _sieve(30)
    
    # Pre-compute check primes for each gap size
    # gap_size -> list of primes < gap_size
    gap_primes_cache = {}
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        g = hi - lo
        
        if g not in gap_primes_cache:
            gap_primes_cache[g] = [p for p in primes if p < g]
        
        check_primes = gap_primes_cache[g]
        
        found = None
        # Iterate through candidates
        for m in range(lo + 1, hi):
            valid = True
            for p in check_primes:
                if m % p == 0:
                    valid = False
                    break
            if valid:
                found = m
                break
                
        result[i] = found
        
    return result
