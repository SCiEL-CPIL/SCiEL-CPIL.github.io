# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of positive integers to reals
    maximizing a = -sum_k f(k) log(k)/k subject to:
      1) After clipping to [-10,10] and adjusting f(1) so sum f(k)/k = 0,
         M(x) = sum_k f(k) floor(x/k) <= 1.0001 for all x in [1, 10K].
      2) K <= 200000, m >= 1.

    Strategy: Use Mobius function mu(k) for k=1..K.
    The identity sum_{k=1}^x mu(k) floor(x/k) = 1 holds for infinite sums.
    For finite K, M(x) = 1 - Tail(x) where Tail(x) >= 0. We verify M(x) <= 1.0001.
    """
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 30.0

    def get_fallback():
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    if time.time() - start_time > time_limit - safety_margin:
        return get_fallback()

    K_max = 200000

    # 1. Compute Mobius function mu(k) for k=1..K_max using linear sieve
    mu = np.zeros(K_max + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(K_max + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    
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
    
    if time.time() - start_time > time_limit - safety_margin:
        return get_fallback()

    # 2. Try different K values and pick the best one
    K_candidates = [5000, 10000, 25000, 50000, 100000, 150000, 200000]
    best_K = None
    best_a = -np.inf
    best_domain = None
    best_values = None

    for K in K_candidates:
        if time.time() - start_time > time_limit - safety_margin:
            break
        
        N = 10 * K
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = mu[1:K+1].astype(np.float64)
        
        # Clip values
        values_clipped = np.clip(values, -10.0, 10.0)
        
        # Adjust f(1) so that sum f(k)/k = 0
        S = np.sum(values_clipped / domain)
        values_adj = values_clipped.copy()
        values_adj[0] -= S
        
        # Verify M(x) <= 1.0001
        g = np.zeros(N + 1, dtype=np.float64)
        
        for k in range(1, K + 1):
            f_k = values_adj[k - 1]
            if f_k != 0.0:
                g[k::k] += f_k
            
            if k % 5000 == 0:
                if time.time() - start_time > time_limit - safety_margin:
                    return get_fallback()
        
        M = np.cumsum(g)
        max_M = np.max(M[1:N+1])
        
        if max_M > 1.0001:
            continue
        
        # Compute metric a = -sum_k f(k) log(k) / k
        log_k = np.log(domain)
        a = -np.sum(values_adj * log_k / domain)
        
        if a > best_a:
            best_a = a
            best_K = K
            best_domain = domain
            best_values = values
    
    if best_domain is None:
        return get_fallback()
    
    return best_domain, best_values

# EVOLVE-BLOCK-END
