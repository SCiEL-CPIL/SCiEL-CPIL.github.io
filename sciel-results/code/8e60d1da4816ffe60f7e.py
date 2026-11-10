import numpy as np
import time

def prime_number_theorem_partial_function():
    start_time = time.time()
    deadline = start_time + 580  # 600s limit, keep margin
    
    # Maximum K allowed by task, but we cap for safety/speed
    K_max = 200000
    K_cap = 50000
    
    # 1. Compute Mobius function mu(k) for k <= K_max
    mu = np.zeros(K_max + 1, dtype=np.int64)
    mu[1] = 1
    primes = []
    is_prime = np.ones(K_max + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    # Linear Sieve for Mobius
    for i in range(2, K_max + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > K_max:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
        
        if time.time() > deadline - 120:
             break

    # 2. Compute prefix sums
    # We only need up to K_cap to keep it safe, but we can compute up to K_max if fast enough
    # Let's compute up to K_max
    inv_k = 1.0 / np.arange(1, K_max + 1)
    log_k = np.log(np.arange(1, K_max + 1))
    mu_k = mu[1:K_max + 1]
    
    S_vals = np.cumsum(mu_k * inv_k)
    A_vals = np.cumsum(mu_k * log_k * inv_k)
    
    # 3. Select best K
    # Condition: S_K >= 0
    # Objective: Maximize a = -A_K (so minimize A_K)
    # Safety: Prefer smaller K if A_K is similar, or just cap K.
    
    # We want the K that minimizes A_K among S_K >= 0.
    # Since A_K approaches -1, smaller A_K is better (more negative).
    # So we look for the most negative A_K.
    
    valid_mask = (S_vals >= 0) & (np.arange(1, K_max + 1) <= K_cap)
    if not np.any(valid_mask):
        # Fallback to seed if no valid K found in cap
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    valid_indices = np.where(valid_mask)[0]
    # Sort by A_K ascending (most negative first)
    sorted_indices = valid_indices[np.argsort(A_vals[valid_indices])]
    
    best_idx = sorted_indices[0]
    K = best_idx + 1
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:K + 1].astype(np.float64)
    
    return domain, values
