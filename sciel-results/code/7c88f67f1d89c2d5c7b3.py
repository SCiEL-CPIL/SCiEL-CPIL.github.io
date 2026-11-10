import numpy as np

def construct():
    N = 123457
    
    # 1. Sieve of Eratosthenes to find primes up to N
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i:N+1:i] = False
            
    primes = np.where(is_prime)[0]
    
    # 2. Compute Largest Prime Factor (LPF) for all numbers up to N
    # Initialize with 1
    lpf = np.ones(N + 1, dtype=np.int64)
    
    # For each prime p, update multiples of p
    # Since we iterate p in increasing order, the last assignment is the largest prime factor
    for p in primes:
        # Multiples of p are p, 2p, 3p, ...
        # We can use slicing: lpf[p::p] = p
        lpf[p::p] = p
        
    # 3. Find the split a + b = N that minimizes max(P(a), P(b))
    # a ranges from 1 to N-1
    a = np.arange(1, N, dtype=np.int64)
    b = N - a
    
    # Compute cost for each a
    # lpf[a] gives P(a), lpf[b] gives P(b)
    cost = np.maximum(lpf[a], lpf[b])
    
    # Find the index of the minimum cost
    min_idx = np.argmin(cost)
    
    best_a = a[min_idx]
    best_b = b[min_idx]
    
    return (int(best_a), int(best_b))
