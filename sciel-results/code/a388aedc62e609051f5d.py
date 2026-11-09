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

    Strategy: Use Mobius function mu(k) for k=1..K, scaled by optimal factor s
    for k>1 to maximize the metric while keeping M(x) <= 1.0001.
    """
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 30.0

    K = 50000
    x_max = 10 * K

    # 1. Compute Mobius function mu(k) for k=1..K using a linear sieve
    mu = np.zeros(K + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
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

    # Domain is 1..K
    domain = np.arange(1, K + 1, dtype=np.int64)
    # Values are mu(k)
    values = mu[1:].astype(np.float64)
    values_clipped = np.clip(values, -10.0, 10.0)

    # T = sum_{k=2}^K mu(k)/k (negative)
    T = np.sum(values_clipped[1:] / domain[1:])

    # Compute M_mu(x) = sum_{k=1}^K mu(k)*floor(x/k) for x in [1, x_max]
    if time.time() - start_time > time_limit - safety_margin:
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    diff_mu = np.zeros(x_max + 1, dtype=np.float64)
    for k in range(1, K + 1):
        f_k = values_clipped[k - 1]
        if f_k == 0:
            continue
        diff_mu[k:x_max+1:k] += f_k
        if k % 10000 == 0 and time.time() - start_time > time_limit - safety_margin:
            domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain_small, values_small

    M_mu = np.cumsum(diff_mu)
    x_range = np.arange(1, x_max + 1, dtype=np.float64)
    H = M_mu[1:x_max+1] - (T + 1) * x_range
    max_H = np.max(H)

    if max_H <= 0:
        s = 10.0
    else:
        s = 1.0001 / max_H
        s = min(s, 10.0)

    # Ensure f(1) = -s*T is within [-10, 10]
    s = min(s, 10.0 / max(abs(T), 1e-15))

    # Apply scaling: f(1) = -s*T, f(k) = s*mu(k) for k>1
    values_scaled = values_clipped.copy()
    values_scaled[1:] *= s
    values_scaled[0] = -s * T
    values_scaled = np.clip(values_scaled, -10.0, 10.0)

    # Verify constraint
    diff_final = np.zeros(x_max + 1, dtype=np.float64)
    for k in range(1, K + 1):
        f_k = values_scaled[k - 1]
        if f_k == 0:
            continue
        diff_final[k:x_max+1:k] += f_k
        if k % 10000 == 0 and time.time() - start_time > time_limit - safety_margin:
            domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain_small, values_small

    M_final = np.cumsum(diff_final)
    max_M = np.max(M_final[1:x_max+1])

    if max_M > 1.0001:
        # Fallback to unscaled
        values_adj = values_clipped.copy()
        S = np.sum(values_adj / domain)
        values_adj[0] -= S
        values_adj = np.clip(values_adj, -10.0, 10.0)
        
        diff_fb = np.zeros(x_max + 1, dtype=np.float64)
        for k in range(1, K + 1):
            f_k = values_adj[k - 1]
            if f_k == 0:
                continue
            diff_fb[k:x_max+1:k] += f_k
        M_fb = np.cumsum(diff_fb)
        max_M_fb = np.max(M_fb[1:x_max+1])
        
        if max_M_fb > 1.0001:
            domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain_small, values_small
        
        return domain, values_adj

    return domain, values_scaled

# EVOLVE-BLOCK-END
