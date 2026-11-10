import numpy as np
import time

def construct():
    N = 123457
    deadline = time.time() + 85  # 85s safety margin before 90s kill
    
    # 1. Sieve of Eratosthenes to find primes up to N
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(N)) + 1):
        if is_prime[i]:
            is_prime[i*i:N+1:i] = False
            
    primes = np.where(is_prime)[0]
    
    # 2. Compute Largest Prime Factor (LPF) for all numbers up to N
    lpf = np.ones(N + 1, dtype=np.int32)
    
    for p in primes:
        lpf[p::p] = p
    
    if time.time() > deadline:
        # Fallback: return a reasonable default
        return (1, N - 1)
    
    # 3. Find the split a + b = N that minimizes max(P(a), P(b))
    a = np.arange(1, N, dtype=np.int32)
    b = N - a
    
    cost = np.maximum(lpf[a], lpf[b])
    min_idx = np.argmin(cost)
    
    best_a = int(a[min_idx])
    best_b = int(b[min_idx])
    
    return (best_a, best_b)
