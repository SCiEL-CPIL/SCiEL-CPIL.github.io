# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy:
      1. Use the Mobius function mu(k) for k=1..K.
      2. K = 200,000 (max allowed).
      3. Compute mu(k) via linear sieve.
      4. Return the raw mu(k) values; the evaluator will clip and adjust f(1).
      5. Verify M(x) <= 1.0001 for x in [1, 10K] using an efficient approach.
      6. If valid, return domain and values.
    """
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 10.0

    K = 200000

    # Compute Mobius function mu(k) for k=1..K using linear sieve
    mu = np.zeros(K + 1, dtype=np.int64)
    mu[1] = 1
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
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

    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)

    # The evaluator will:
    # 1. Clip values to [-10, 10]
    # 2. Compute S = sum f(k)/k over clipped values
    # 3. Replace f(1) with f(1) - S
    # 4. Check M(x) = sum f(k) floor(x/k) <= 1.0001 for x in [1, 10K]
    
    # To verify, we simulate the evaluator's steps:
    values_clipped = np.clip(values, -10.0, 10.0)
    S = np.sum(values_clipped / domain)
    values_adj = values_clipped.copy()
    values_adj[0] -= S  # f(1) adjusted

    # Verify M(x) <= 1.0001 for x in [1, 10K]
    # M(x) = sum_{k=1}^K f(k) floor(x/k)
    # Use difference array: M(x) - M(x-1) = sum_{k|x} f(k)
    # So we compute g(j) = sum_{k|j} f(k) for j=1..10K, then M(x) = cumsum(g)
    
    x_max = 10 * K
    g = np.zeros(x_max + 1, dtype=np.float64)
    
    elapsed = time.time() - start_time
    if elapsed > time_limit - safety_margin:
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    # For each k, add f(k) to g[j] for all j that are multiples of k
    for k in range(1, K + 1):
        f_k = values_adj[k - 1]
        if f_k == 0.0:
            continue
        # Add f_k to g at indices k, 2k, 3k, ..., <= x_max
        # Use slice assignment for efficiency
        g[k:x_max+1:k] += f_k
        
        # Periodic time check
        if k % 50000 == 0:
            if time.time() - start_time > time_limit - safety_margin:
                domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
                values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
                return domain_small, values_small

    # M(x) = cumsum of g
    M = np.cumsum(g)
    
    # Check max M(x) for x in [1, x_max]
    max_M = np.max(M[1:x_max+1])
    
    if max_M > 1.0001:
        # Infeasible - fall back to small solution
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    # Return the raw values (evaluator will clip and adjust)
    return domain, values

# EVOLVE-BLOCK-END
