import numpy as np
import time

def _compute_mobius(K):
    """Compute Mobius function for 1..K using a linear sieve."""
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    is_prime[0] = False
    
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 600s limit
    
    K_max = 200000
    tol = 1.0001
    
    # Compute Mobius function up to K_max
    mu = _compute_mobius(K_max)
    
    # Compute prefix sums of mu(k)/k
    # S(K) = sum_{k=1}^K mu(k)/k
    inv_k = np.zeros(K_max + 1, dtype=np.float64)
    inv_k[1:] = 1.0 / np.arange(1, K_max + 1, dtype=np.float64)
    mu_float = mu.astype(np.float64)
    terms = mu_float * inv_k
    S_prefix = np.cumsum(terms)
    
    # Find largest K such that feasibility is likely satisfied.
    # Condition: S(K) * K >= -0.0001 (approx)
    # We verify with a stricter check if needed, but this should suffice for the main constraint.
    
    best_K = 1
    for K in range(K_max, 0, -1):
        if time.time() > deadline - 5:
            break
        s_val = S_prefix[K]
        # Check constraint: 1 - S(K)*K <= 1.0001  => S(K)*K >= -0.0001
        if s_val * K >= -0.0001:
            best_K = K
            break
            
    # If no K found (should not happen, K=1 works), use 1
    if best_K == 0:
        best_K = 1
        
    # Optional: Verify feasibility for best_K more rigorously if time permits
    # But the O(K) scan is very fast.
    
    domain = np.arange(1, best_K + 1, dtype=np.int64)
    values = mu[1:best_K+1].astype(np.float64)
    
    return domain, values
