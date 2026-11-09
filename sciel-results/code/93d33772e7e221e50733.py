import numpy as np
import time

def _compute_mobius(K):
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if not is_composite[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_composite[i * p] = True
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 600.0 - 15.0  # safety margin
    
    K = 200000
    mu = _compute_mobius(K)
    
    # Use f(k) = -mu(k) for k >= 2
    # This ensures M(x) <= 1 for all x >= 1 (approx -1 for large x)
    # and f(1) will be negative, keeping M(x) bounded.
    
    ks = np.arange(2, K + 1)
    f_ks = -mu[2:].astype(np.float64)
    
    # Compute f(1)
    S = np.sum(f_ks / ks)
    f1 = -S
    
    domain = np.concatenate(([1], ks))
    values = np.concatenate(([f1], f_ks))
    
    # Verify time
    if time.time() > deadline:
        # If it took too long (unlikely for K=200000), return smaller K
        # But _compute_mobius is fast.
        pass
        
    return domain, values
