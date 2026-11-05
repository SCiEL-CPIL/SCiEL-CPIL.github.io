"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

# EVOLVE-BLOCK-START
import time

def _primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0: sieve[0] = False
    if limit >= 1: sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 280  # 280s safety margin within 330s limit
    
    S = list(range(N + 1))
    primes = _primes(N)
    
    while True:
        if time.time() > deadline:
            break
        
        L = len(S)
        if L == 0:
            break
            
        # Find all violating primes and their min class sizes
        best_p = None
        best_min_size = float('inf')
        best_residue = None
        
        for p in primes:
            if p > L:
                break
            
            counts = [0] * p
            for x in S:
                counts[x % p] += 1
            
            min_c = min(counts)
            if min_c > 0:
                # This prime is violating
                if min_c < best_min_size:
                    best_min_size = min_c
                    best_p = p
                    best_residue = counts.index(min_c)
        
        if best_p is None:
            break  # S is admissible
        
        # Remove all x in S such that x % best_p == best_residue
        S = [x for x in S if x % best_p != best_residue]
        
        if len(S) == L:
            break

    return S
# EVOLVE-BLOCK-END
