import numpy as np

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    # Start with all integers in [0, N]
    S = list(range(N + 1))
    primes = _primes(N)
    
    i = 0
    while i < len(primes):
        p = primes[i]
        if p > len(S):
            break
            
        # Count residues mod p
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
            
        min_count = min(counts)
        
        if min_count == 0:
            # Already admissible for p
            i += 1
        else:
            # Find the residue with minimum count
            r = counts.index(min_count)
            # Remove all elements congruent to r mod p
            S = [x for x in S if x % p != r]
            # Do not increment i, re-check p with new S
            if not S:
                break
                
    return S
