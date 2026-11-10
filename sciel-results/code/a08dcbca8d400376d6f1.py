# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600s margin

    # Maximum allowed key
    K_max = 200000

    # We try to use the largest K possible.
    K = K_max

    # 1. Compute Mobius function mu(k) for k <= K
    mu = np.ones(K + 1, dtype=np.int64)
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    
    # Linear sieve for Mobius
    mu[1] = 1
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

    # Time check
    if time.time() > deadline - 60:
        return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

    # 2. Check feasibility
    # M(x) = sum_{k=1}^K mu(k) floor(x/k)
    # For x <= K, M(x) = 1.
    # For x > K, M(x) = 1 + sum_{j=K+1}^x g(j), where g(j) = sum_{d|j, d<=K} mu(d).
    # We need max_{x in [K+1, 10K]} M(x) <= 1.0001.
    # This is equivalent to max_{x in [K+1, 10K]} sum_{j=K+1}^x g(j) <= 0.0001.
    
    # Compute g(j) for j in [1, 10K]
    # g(j) = sum_{d|j, d<=K} mu(d)
    # We can compute this by iterating d from 1 to K and adding mu(d) to g(m) for m = d, 2d, ... <= 10K.
    
    max_x = 10 * K
    g = np.zeros(max_x + 1, dtype=np.float64)
    
    # Efficient update of g
    # Using NumPy slicing for speed
    for d in range(1, K + 1):
        if mu[d] != 0:
            g[d:max_x+1:d] += mu[d]
            
    # Compute prefix sum of g(j) for j > K
    # We only care about j from K+1 to 10K.
    # Let S(x) = sum_{j=K+1}^x g(j).
    # We need max S(x) for x in [K+1, 10K].
    
    if time.time() > deadline - 30:
         # Fallback
         return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

    g_tail = g[K+1:max_x+1]
    S = np.cumsum(g_tail)
    
    max_S = np.max(S)
    
    # Feasibility condition: M(x) <= 1.0001
    # M(x) = 1 + S(x)
    # So 1 + S(x) <= 1.0001 => S(x) <= 0.0001
    
    if max_S <= 0.0001:
        # Feasible with K
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = mu[1:K+1].astype(np.float64)
        return domain, values
    else:
        # Infeasible with K.
        # We need to find a smaller K or modify f.
        # Modifying f is hard. Let's try a smaller K.
        # But reducing K reduces the score.
        # Let's try to find the largest feasible K.
        # We can binary search K.
        
        # However, computing mu and g for different K is expensive.
        # Let's just try K = 100000, 50000, etc.
        # Or, we can observe that if K is large, the oscillations of S(x) might be small.
        # If K=200000 is infeasible, maybe K=100000 is feasible.
        
        # Let's try a heuristic: reduce K until feasible.
        # But we already computed mu for K=200000.
        # We can reuse mu.
        # We need to recompute g for the new K.
        
        # Let's try K = 100000
        K_try = 100000
        max_x_try = 10 * K_try
        g_try = np.zeros(max_x_try + 1, dtype=np.float64)
        
        for d in range(1, K_try + 1):
            if mu[d] != 0:
                g_try[d:max_x_try+1:d] += mu[d]
                
        g_tail_try = g_try[K_try+1:max_x_try+1]
        S_try = np.cumsum(g_tail_try)
        max_S_try = np.max(S_try)
        
        if max_S_try <= 0.0001:
            domain = np.arange(1, K_try + 1, dtype=np.int64)
            values = mu[1:K_try+1].astype(np.float64)
            return domain, values
        else:
            # Try K = 50000
            K_try = 50000
            max_x_try = 10 * K_try
            g_try = np.zeros(max_x_try + 1, dtype=np.float64)
            
            for d in range(1, K_try + 1):
                if mu[d] != 0:
                    g_try[d:max_x_try+1:d] += mu[d]
                    
            g_tail_try = g_try[K_try+1:max_x_try+1]
            S_try = np.cumsum(g_tail_try)
            max_S_try = np.max(S_try)
            
            if max_S_try <= 0.0001:
                domain = np.arange(1, K_try + 1, dtype=np.int64)
                values = mu[1:K_try+1].astype(np.float64)
                return domain, values
            else:
                # Fallback to the small baseline
                return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

# EVOLVE-BLOCK-END
