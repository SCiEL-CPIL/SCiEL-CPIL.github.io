import math
import random

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes up to N, as K will be <= N
PRIMES = _primes(N)

def _sieve_with_strategy(K, strategy):
    valid_indices = list(range(N + 1))
    relevant_primes = [p for p in PRIMES if p <= K]
    
    for p in relevant_primes:
        if len(valid_indices) < K:
            return []
        
        counts = [0] * p
        for v in valid_indices:
            counts[v % p] += 1
        
        min_count = min(counts)
        
        # Find all residues with min count
        candidates = [r for r, c in enumerate(counts) if c == min_count]
        
        if strategy == 0:
            min_res = candidates[0]
        elif strategy == 1:
            min_res = candidates[-1]
        else:
            min_res = random.choice(candidates)
            
        valid_indices = [v for v in valid_indices if v % p != min_res]
        
    if len(valid_indices) >= K:
        return valid_indices[:K]
    return []

def construct():
    # Estimate upper bound
    # K ln K ~ 2807 -> K ~ 460
    # Start from 500 to be safe.
    
    best_k = 0
    best_set = []
    
    # Try K from 500 down to 1
    for K in range(500, 0, -1):
        found = False
        # Try a few strategies
        # Strategy 0: First min
        # Strategy 1: Last min
        # Strategy 2: Random min (with fixed seed for reproducibility? No, random is fine if we just need *a* solution)
        # But we want the *largest* K. 
        # If K works with *any* strategy, we accept it.
        
        for strat in [0, 1, 2]:
            res = _sieve_with_strategy(K, strat)
            if res:
                best_k = K
                best_set = res
                found = True
                break
        
        if found:
            break
            
    return best_set
