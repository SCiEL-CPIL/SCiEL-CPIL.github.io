# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f approximating the Mobius function to maximize the objective.
    """
    start_time = time.time()
    deadline = 630.0
    
    # Try K values from large to small, checking time
    for K in [150000, 100000, 75000, 50000, 30000, 20000, 10000, 5000]:
        if time.time() > deadline - 120:
            break
        
        # 1. Compute Mobius function up to K
        mu = np.ones(K + 1, dtype=np.float64)
        is_prime = np.ones(K + 1, dtype=bool)
        
        mu[0] = 0
        for i in range(2, K + 1):
            if is_prime[i]:
                for j in range(i, K + 1, i):
                    is_prime[j] = False
                    mu[j] *= -1
                i2 = i * i
                if i2 <= K:
                    for j in range(i2, K + 1, i2):
                        mu[j] = 0

        # 2. Compute S = sum_k mu(k)/k for k=1..K
        # mu(1) = 1
        S = np.sum(mu[1:K+1] / np.arange(1, K+1, dtype=np.float64))
        
        # Adjusted values: f(1) = 1 - S, f(k) = mu(k) for k >= 2
        values = mu[1:K+1].copy()
        values[0] = 1.0 - S  # f(1) = mu(1) - S = 1 - S
        
        # 3. Check feasibility: M(x) = sum_k f(k) * floor(x/k) for x in [1, 10K]
        # M(x) = sum_{k=1}^{K} f(k) * floor(x/k)
        # = sum_{k=1}^{K} f(k) * sum_{n: k|n, n<=x} 1
        # = sum_{n=1}^{x} sum_{k|n, k<=K} f(k)
        # Let A(n) = sum_{k|n, k<=K} f(k), then M(x) = sum_{n=1}^{x} A(n)
        
        max_x = 10 * K
        A = np.zeros(max_x + 1, dtype=np.float64)
        
        # For each k, add f(k) to A[k], A[2k], A[3k], ...
        for k in range(1, K + 1):
            if values[k-1] != 0:
                A[k:max_x+1:k] += values[k-1]
            
            if time.time() > deadline - 60:
                break
        
        if time.time() > deadline - 60:
            continue
        
        M = np.cumsum(A)
        
        if np.max(M) > 1.0001:
            continue
        
        # 4. Compute metric: a = -sum_k f(k) * log(k) / k
        ks = np.arange(1, K + 1, dtype=np.float64)
        a = -np.sum(values * np.log(ks) / ks)
        
        if time.time() > deadline - 30:
            break
        
        domain = np.arange(1, K + 1, dtype=np.int64)
        return domain, values
    
    # Fallback: small K
    K = 1000
    mu = np.ones(K + 1, dtype=np.float64)
    is_prime = np.ones(K + 1, dtype=bool)
    mu[0] = 0
    for i in range(2, K + 1):
        if is_prime[i]:
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            i2 = i * i
            if i2 <= K:
                for j in range(i2, K + 1, i2):
                    mu[j] = 0
    
    S = np.sum(mu[1:K+1] / np.arange(1, K+1, dtype=np.float64))
    values = mu[1:K+1].copy()
    values[0] = 1.0 - S
    
    domain = np.arange(1, K + 1, dtype=np.int64)
    return domain, values

# EVOLVE-BLOCK-END
