import math
from math import isqrt

def construct():
    # Targeted generation of highly composite / smooth k values.
    # Strategy:
    # 1. Generate all k <= 10^9 that are "highly smooth" (primarily 2,3,5,7 with some 11,13).
    # 2. Compute sigma(k) efficiently using the factorization inherent in generation.
    # 3. Compute M = k * sigma(k).
    # 4. Group by M and find the M with the most distinct k.
    
    # Primes to include. More primes = more numbers, but sigma grows faster.
    # Let's stick to a manageable set.
    primes = [2, 3, 5, 7, 11, 13, 17, 19]
    
    max_k = 10**9
    solutions = {} # M -> list of k
    
    # Recursive generation
    # state: (current_k, current_sigma, index_in_primes)
    # We want to generate all k = prod p_i^a_i
    # sigma(k) = prod (p_i^(a_i+1)-1)/(p_i-1)
    
    def gen(idx, k, sigma_k):
        if k > max_k:
            return
        
        # Add this k to our tracking
        M = k * sigma_k
        if M <= 10**18:
            if M not in solutions:
                solutions[M] = []
            solutions[M].append(k)
        
        if idx == len(primes):
            return
        
        p = primes[idx]
        # Try powers of p
        pk = 1
        sigma_p = 1
        # p^0 = 1, sigma contribution 1
        # p^1 = p, sigma contribution 1+p
        # ...
        while True:
            next_k = k * pk
            if next_k > max_k:
                break
            
            # Calculate sigma(p^a) = 1 + p + ... + p^a
            # We are multiplying by p each step.
            # If current power is a, sigma_p_new = sigma_p_old + p^a
            # Actually, let's just compute it directly or incrementally.
            
            # Incremental sigma for prime power:
            # If we go from p^a to p^(a+1), sigma increases by p^(a+1).
            # Start: a=0, pk=1, sigma_p=1
            # Next: a=1, pk=p, sigma_p = 1+p
            # Next: a=2, pk=p^2, sigma_p = 1+p+p^2
            
            # Recurse with next prime index
            gen(idx + 1, next_k, sigma_k * sigma_p)
            
            # Prepare for next power
            pk *= p
            if pk > max_k // k:
                break
            sigma_p += pk
        
    # Start generation
    gen(0, 1, 1)
    
    # Find best M
    best_M = 1
    best_k = [1]
    best_count = 1
    
    for M, ks in solutions.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_k = ks
            
    # Ensure types are correct
    return {'M': int(best_M), 'k': [int(x) for x in best_k]}

