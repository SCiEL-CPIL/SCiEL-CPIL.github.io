# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = start_time + 590.0  # 10s safety margin

    K_max = 200000
    K = K_max

    # 1. Compute Mobius function mu(k) for k <= K
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

    def check_feasibility(K_val, mu_arr):
        """Check if f(k)=mu(k) for k<=K_val is feasible."""
        max_x = 10 * K_val
        g = np.zeros(max_x + 1, dtype=np.float64)
        
        # Only iterate over d where mu[d] != 0
        nonzero_d = np.nonzero(mu_arr[1:K_val+1])[0] + 1
        for d in nonzero_d:
            g[d:max_x+1:d] += mu_arr[d]
            
        g_tail = g[K_val+1:max_x+1]
        S = np.cumsum(g_tail)
        max_S = np.max(S)
        return max_S <= 0.0001

    # Try K values in decreasing order
    K_candidates = [200000, 150000, 100000, 75000, 50000, 30000, 20000, 10000, 5000]
    
    for K_try in K_candidates:
        if time.time() > deadline - 30:
            break
        if check_feasibility(K_try, mu):
            domain = np.arange(1, K_try + 1, dtype=np.int64)
            values = mu[1:K_try+1].astype(np.float64)
            return domain, values

    # Fallback to small baseline
    return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

# EVOLVE-BLOCK-END
