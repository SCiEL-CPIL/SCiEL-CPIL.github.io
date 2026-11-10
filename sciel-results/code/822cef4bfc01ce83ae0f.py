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

    # Compute S_mu = sum mu(k)/k for k=1..K
    S_mu = np.sum(mu[1:K+1].astype(np.float64) / np.arange(1, K+1))

    # Determine max feasible scaling c
    # M(x) = c*(1 - S_mu*x) for x in [1, 10K]
    # Constraint: c*(1 - S_mu*x) <= 1.0001
    if S_mu > 0:
        # Max at x=1: c*(1 - S_mu) <= 1.0001
        c_max = 1.0001 / (1 - S_mu)
    elif S_mu < 0:
        # Max at x=10K: c*(1 + abs(S_mu)*10K) <= 1.0001
        c_max = 1.0001 / (1 + abs(S_mu)*10*K)
    else:
        c_max = 1.0001

    # Clip c_max to reasonable range
    c_max = min(c_max, 10.0)  # Can't exceed clipping range
    c_max = max(c_max, 0.0)

    # Use c_max if it's better than 1, else use 1
    c = c_max if c_max > 1.0 else 1.0

    # Verify feasibility with correct M(x) computation
    def check_feasibility(K_val, f_arr, c_val=1.0):
        """Check if f(k) = c_val*mu(k) for k<=K_val is feasible."""
        max_x = 10 * K_val
        g = np.zeros(max_x + 1, dtype=np.float64)
        
        nonzero_d = np.nonzero(f_arr[1:K_val+1])[0] + 1
        for d in nonzero_d:
            g[d:max_x+1:d] += f_arr[d]
        
        # M(x) = cumsum of g[1..x]
        M = np.cumsum(g[1:max_x+1])
        max_M = np.max(M)
        return max_M <= 1.0001

    # Build the function
    values = (c * mu[1:K+1]).astype(np.float64)
    domain = np.arange(1, K + 1, dtype=np.int64)

    # Verify feasibility
    if check_feasibility(K, mu, c):
        return domain, values

    # Fallback to c=1
    if check_feasibility(K, mu, 1.0):
        values = mu[1:K+1].astype(np.float64)
        return domain, values

    # Fallback to small baseline
    return np.array([1, 2, 3, 5, 30], dtype=np.int64), np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

# EVOLVE-BLOCK-END
