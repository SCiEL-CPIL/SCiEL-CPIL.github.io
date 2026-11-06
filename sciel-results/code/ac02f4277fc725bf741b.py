import time
import math

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
    start_time = time.time()
    deadline = start_time + 280  # 300s limit, keep margin
    
    # Initial set
    S = list(range(N + 1))
    k = len(S)
    
    # All primes up to N
    all_primes = _primes(N)
    
    # Active primes: those <= k
    # We will maintain a list of active primes
    active_primes = [p for p in all_primes if p <= k]
    
    # To speed up, we can precompute or just calculate on the fly
    # Given N=5000, on the fly is fine.
    
    while active_primes:
        if time.time() > deadline:
            break
            
        # Find the prime with the minimum cost (min residue count)
        # Cost is the number of elements to remove to satisfy this prime
        best_p = None
        best_r = None
        min_count = float('inf')
        
        for p in active_primes:
            # Count residues
            counts = [0] * p
            for x in S:
                counts[x % p] += 1
            
            # Check if already satisfied (some count is 0)
            # If min(counts) == 0, it's satisfied.
            # But we are iterating to find the one to *process*.
            # If a prime is already satisfied, we should remove it from active.
            # However, we are looking for the one to *act on*.
            # If all active primes are satisfied, we are done.
            
            c_min = min(counts)
            
            if c_min == 0:
                # This prime is already satisfied.
                # We can mark it for removal.
                # But we need to check all to find the "worst" unsatisfied one?
                # Actually, if it's satisfied, it's not a constraint.
                # We just need to ensure all active primes are satisfied.
                # So if we find one with c_min > 0, it's violated.
                # We want to fix the violated ones.
                # If ALL have c_min == 0, we break.
                continue
                
            # If c_min > 0, it is violated.
            # We compare c_min with min_count.
            if c_min < min_count:
                min_count = c_min
                best_p = p
                # Find the residue with min count
                best_r = counts.index(c_min)
                
        if best_p is None:
            # All active primes are satisfied
            break
            
        # Remove elements from S corresponding to best_r mod best_p
        # S = [x for x in S if x % best_p != best_r]
        
        # Optimization: Use a set or boolean mask?
        # List comprehension is fine for 5000.
        S = [x for x in S if x % best_p != best_r]
        k = len(S)
        
        # Update active primes
        # 1. Remove best_p
        # 2. Remove any p > k
        active_primes = [p for p in active_primes if p != best_p and p <= k]
        
    return S
