import numpy as np
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = 85.0 # 5s margin before 90s limit
    
    # 1. Sieve of Eratosthenes to find primes up to N
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i:N+1:i] = False
            
    primes = np.where(is_prime)[0]
    
    # 2. Compute Largest Prime Factor (LPF) for all numbers up to N
    # Initialize with 1
    # Use int32 as N < 2^31, saving memory and potentially improving cache locality
    lpf = np.ones(N + 1, dtype=np.int32)
    
    # For each prime p, update multiples of p
    # Since we iterate p in increasing order, the last assignment is the largest prime factor
    for p in primes:
        # Check deadline periodically? 
        # For N=123457, this loop is very fast (<0.1s).
        # We add a check to satisfy the directive.
        if time.time() > deadline:
            # Fallback: Return a trivial split if we run out of time
            # This should never happen for this N.
            return (1, N-1)
        lpf[p::p] = p
        
    # 3. Find the split a + b = N that minimizes max(P(a), P(b))
    # Use symmetry: a and b are interchangeable. We only need to check a <= N/2.
    half_N = N // 2
    a = np.arange(1, half_N + 1, dtype=np.int32)
    b = N - a
    
    # Compute cost for each a
    # lpf[a] gives P(a), lpf[b] gives P(b)
    cost = np.maximum(lpf[a], lpf[b])
    
    # Find the index of the minimum cost
    min_idx = np.argmin(cost)
    
    best_a = a[min_idx]
    best_b = b[min_idx]
    
    return (int(best_a), int(best_b))
