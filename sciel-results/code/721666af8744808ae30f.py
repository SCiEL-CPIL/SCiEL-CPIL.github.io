# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy:
      The objective is a = -sum_k f(k) log(k)/k, subject to:
        1) After clipping to [-10,10] and adjusting f(1) so sum_k f(k)/k = 0,
           M(x) = sum_k f(k) floor(x/k) <= 1.0001 for all x in [1, 10K].
        2) K <= 200000, m >= 1.

      The ideal candidate is the Mobius function mu(k), for which:
        sum_k mu(k) floor(x/k) = 1 for all x >= 1 (exact identity),
        and -sum_k mu(k) log(k)/k converges to 1 as K -> infinity.

      We compute mu(k) for k=1..K using a sieve. Since the sum of mu(k)/k over all k
      is 0 in the limit (and the partial sums are small), the adjustment S = sum f(k)/k
      is small, and f(1) = -S keeps the constraint M(x) <= 1 approximately.

      Verification is done in O(K log K) time using a difference array for the 
      divisor sum g(j) = sum_{k|j} f(k), then prefix sum for M(x).
      This is faster than the naive O(K^2) approach.
    """
    start_time = time.time()
    time_limit = 630.0
    safety_margin = 10.0

    # Maximum allowed K
    K = 200000

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

    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)

    # 2. Clip values to [-10, 10]
    values_clipped = np.clip(values, -10.0, 10.0)

    # 3. Adjust f(1)
    # S = sum f(k)/k
    S = np.sum(values_clipped / domain)
    values_adj = values_clipped.copy()
    values_adj[0] -= S  # f(1) is at index 0

    # 4. Verify M(x) <= 1.0001
    x_max = 10 * K
    
    # Check time before heavy computation
    if time.time() - start_time > time_limit - safety_margin:
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    # diff array for M(x)
    # M(x) = sum_{j=1}^x sum_{k|j} f(k)
    # Let g(j) = sum_{k|j} f(k). Then M(x) = cumsum(g)[x].
    # We can compute g(j) by iterating k and adding f(k) to g(j) for all j multiple of k.
    # This is equivalent to adding f(k) to diff[j*k] for j=1,2,...
    
    diff = np.zeros(x_max + 2, dtype=np.float64)
    
    # Vectorized update:
    # For each k, indices are k, 2k, ..., floor(x_max/k)*k.
    # We can do this in a loop over k, which is K iterations.
    # Inside, we do a vector addition to diff[k::k] up to x_max.
    
    for k in range(1, K + 1):
        f_k = values_adj[k - 1]
        if f_k == 0:
            continue
        # Add f_k to diff at indices k, 2k, ..., <= x_max
        # Slice: start=k, stop=x_max+1, step=k
        diff[k:x_max+1:k] += f_k
        
        # Check time periodically
        if (k % 10000 == 0) and (time.time() - start_time > time_limit - safety_margin):
            domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain_small, values_small

    # Compute M(x) = cumsum(diff)
    # diff[0] is 0, so M[0] = 0.
    # M[x] = sum_{j=1}^x diff[j]
    M = np.cumsum(diff)
    
    # Check max M(x) for x in [1, x_max]
    max_M = np.max(M[1:x_max+1])
    
    if max_M > 1.0001:
        # Infeasible. This should not happen for K=200,000 with Mobius.
        # But if it does, we might try a smaller K or return fallback.
        # For now, fallback to be safe if constraint fails unexpectedly.
        domain_small = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values_small = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain_small, values_small

    # 5. Compute metric a
    log_k = np.log(domain)
    a = -np.sum(values_adj * log_k / domain)

    return domain, values

# EVOLVE-BLOCK-END
