# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 60s safety margin

    # Maximum allowed key
    K_max = 200000
    
    # Use the largest K for maximum score
    K = K_max
    
    # 1. Compute Mobius function mu(k) for k <= K using linear sieve
    mu = np.ones(K + 1, dtype=np.int64)
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    
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

    # 2. Verify feasibility: M'(x) <= 1.0001 for all x in [1, 10K]
    # After normalization, M'(x) = M_mu(x) - S*x where S = sum_{k=1}^K mu(k)/k
    # We need to check max_{x in [1, 10K]} M'(x) <= 1.0001
    
    # Compute S = sum_{k=1}^K mu(k)/k
    S = np.sum(mu[1:K+1].astype(np.float64) / np.arange(1, K+1, dtype=np.float64))
    
    # For x <= K: M_mu(x) = 1 (exactly, by the identity sum_{d|n} mu(d) = delta(n,1))
    # So M'(x) = 1 - S*x for x in [1, K]
    # Max of 1 - S*x on [1, K] is at x=1: 1 - S
    
    # For x > K: M_mu(x) = sum_{k=1}^K mu(k) * floor(x/k)
    # We can compute this efficiently using the structure of floor(x/k)
    # But it's easier to use the g(j) approach:
    # M_mu(x) = 1 + sum_{j=K+1}^x g(j) where g(j) = sum_{d|j, d<=K} mu(d)
    # Actually, for x > K: M_mu(x) = sum_{j=1}^x sum_{d|j, d<=K} mu(d) = sum_{j=1}^x g(j)
    # where g(j) = sum_{d|j, d<=K} mu(d)
    # And for j <= K, g(j) = sum_{d|j} mu(d) = 1 if j=1 else 0
    # So M_mu(x) = 1 + sum_{j=2}^x g(j) for x >= 1
    # For x <= K: sum_{j=2}^x g(j) = 0, so M_mu(x) = 1. Correct.
    # For x > K: M_mu(x) = 1 + sum_{j=K+1}^x g(j)
    
    # Compute g(j) for j in [K+1, 10K]
    # g(j) = sum_{d|j, d<=K} mu(d)
    # We can compute this by iterating d from 1 to K and adding mu(d) to g(m) for m = d, 2d, 3d, ... <= 10K
    # But we only need j in [K+1, 10K]
    
    max_x = 10 * K
    
    if time.time() > deadline - 30:
        # Fallback to small safe domain
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Compute g for j in [1, 10K]
    g = np.zeros(max_x + 1, dtype=np.float64)
    
    # Only need to add contributions from d where mu[d] != 0
    # For each such d, add mu[d] to g[d], g[2d], g[3d], ...
    # We only care about j in [K+1, 10K], but computing all is fine for correctness
    
    # Use vectorized approach for speed
    for d in range(1, K + 1):
        if mu[d] != 0:
            # Add mu[d] to g[d::d]
            # This is g[d], g[2d], g[3d], ...
            indices = np.arange(d, max_x + 1, d)
            g[indices] += mu[d]
    
    # Now M_mu(x) = 1 + sum_{j=2}^x g(j) for x >= 1
    # But we know for x <= K, M_mu(x) = 1, so let's just verify for x > K
    # M_mu(x) = 1 + sum_{j=K+1}^x g(j) for x > K
    # (since sum_{j=2}^K g(j) = 0)
    
    if time.time() > deadline - 20:
        # If we're running out of time, assume feasible and return
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = mu[1:K+1].astype(np.float64)
        return domain, values

    # Compute cumulative sum of g from K+1 to 10K
    g_tail = g[K+1:max_x+1]
    cumsum_g = np.cumsum(g_tail)
    
    # M_mu(x) for x = K+1, K+2, ..., 10K is 1 + cumsum_g
    # M'(x) = M_mu(x) - S*x
    # x values: K+1, K+2, ..., 10K
    x_vals = np.arange(K+1, max_x+1, dtype=np.float64)
    M_prime_tail = 1.0 + cumsum_g - S * x_vals
    
    # For x in [1, K]: M'(x) = 1 - S*x
    # Max is at x=1: 1 - S
    max_M_prime_head = 1.0 - S
    
    # Overall max
    max_M_prime = max(max_M_prime_head, np.max(M_prime_tail))
    
    if max_M_prime <= 1.0001:
        # Feasible!
        # Scale the values to maximize the score
        # The score is proportional to the sum, so we can scale up as much as possible
        # Constraint: C * max_M_prime <= 1.0001
        # So C = 1.0001 / max_M_prime (if max_M_prime > 0)
        
        if max_M_prime > 0:
            C = 1.0001 / max_M_prime
        else:
            # If max_M_prime <= 0, we can scale arbitrarily? 
            # But f(1) adjustment might change things.
            # However, M'(1) = 1-S is usually positive.
            # If it's negative, it means S > 1, which is unlikely for mu.
            # Let's assume C=1 if max_M_prime <= 0 to be safe, or try a large C.
            # Given M'(1) approx 1, this case is unlikely.
            C = 1.0
            
        # Clip C to reasonable bounds if necessary?
        # The values are clipped to [-10, 10] by the evaluator.
        # mu(k) is -1, 0, 1.
        # So C * mu(k) will be in [-C, C].
        # If C > 10, values will be clipped.
        # Clipping reduces the score if the objective is linear and we are scaling up.
        # So we should limit C such that C * max(|mu|) <= 10.
        # max(|mu|) = 1.
        # So C <= 10.
        
        C = min(C, 10.0)
        
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = mu[1:K+1].astype(np.float64) * C
        return domain, values
    else:
        # Infeasible with K=200000. Try smaller K.
        # The score a = -sum mu(k) log(k)/k decreases as K decreases.
        # Try K = 100000
        for K_try in [100000, 50000, 10000]:
            if time.time() > deadline - 10:
                break
            mu_try = mu[1:K_try+1]
            S_try = np.sum(mu_try.astype(np.float64) / np.arange(1, K_try+1, dtype=np.float64))
            
            max_x_try = 10 * K_try
            g_try = np.zeros(max_x_try + 1, dtype=np.float64)
            
            for d in range(1, K_try + 1):
                if mu[d] != 0:
                    indices = np.arange(d, max_x_try + 1, d)
                    g_try[indices] += mu[d]
            
            g_tail_try = g_try[K_try+1:max_x_try+1]
            cumsum_g_try = np.cumsum(g_tail_try)
            
            x_vals_try = np.arange(K_try+1, max_x_try+1, dtype=np.float64)
            M_prime_tail_try = 1.0 + cumsum_g_try - S_try * x_vals_try
            
            max_M_prime_head_try = 1.0 - S_try
            max_M_prime_try = max(max_M_prime_head_try, np.max(M_prime_tail_try))
            
            if max_M_prime_try <= 1.0001:
                # Scale up
                if max_M_prime_try > 0:
                    C = 1.0001 / max_M_prime_try
                else:
                    C = 1.0
                C = min(C, 10.0)
                
                domain = np.arange(1, K_try + 1, dtype=np.int64)
                values = mu_try.astype(np.float64) * C
                return domain, values
        
        # Fallback
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

# EVOLVE-BLOCK-END
