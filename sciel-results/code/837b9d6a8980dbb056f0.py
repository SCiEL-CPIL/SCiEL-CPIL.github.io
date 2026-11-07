import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

PRIMES = sieve(100)

# Pre-compute relevant primes for each possible gap size
MAX_GAP = max(hi - lo for lo, hi in P['gaps'])
RELEVANT_PRIMES = {}
for g in range(2, MAX_GAP + 1):
    RELEVANT_PRIMES[g] = [p for p in PRIMES if p < g]

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        
        # Special case: gap=2, any m in (lo, hi) works
        if gap == 2:
            result[i] = lo + 1
            continue
        
        relevant = RELEVANT_PRIMES[gap]
        
        # Search for witness
        for m in range(lo + 1, hi):
            is_valid = True
            for p in relevant:
                if m % p == 0:
                    is_valid = False
                    break
            
            if is_valid:
                result[i] = m
                break
    
    return result
