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

    Strategy: Use a scaled subset of Mobius function values.
    We focus on k where mu(k) = -1 (square-free with odd number of prime factors),
    setting f(k) = -alpha (a negative constant) for these k, and 0 otherwise.
    This maximizes the objective since -f(k)*log(k)/k = alpha*log(k)/k > 0.
    The constraint sum f(k)/k = 0 is handled by adjusting f(1).
    We scale alpha so that M(x) <= 1.0001 for all x.
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
    K_candidates = [10000, 25000, 50000, 100000, 150000, 200000]
    best_K = None
    best_a = -np.inf
    best_domain = None
    best_values = None

    for K in K_candidates:
        if time.time() - start_time > time_limit - safety_margin:
            break
        
        N = 10 * K
        
        # Indices where mu(k) = -1
        idx_neg = np.where(mu[1:K+1] == -1)[0] + 1  # 1-based indices
        
        # For these indices, f(k) = -alpha (negative)
        # For other indices (except 1), f(k) = 0
        # f(1) will be adjusted
        
        # S = sum f(k)/k = -alpha * sum_{k in idx_neg} 1/k
        # f(1) = -S = alpha * sum_{k in idx_neg} 1/k
        
        # M(x) = f(1)*floor(x/1) + sum_{k in idx_neg} (-alpha) * floor(x/k)
        #      = alpha * sum_{k in idx_neg} 1/k * x - alpha * sum_{k in idx_neg} floor(x/k)
        #      = alpha * [x * sum_{k in idx_neg} 1/k - sum_{k in idx_neg} floor(x/k)]
        
        # For M(x) <= 1.0001, we need:
        # alpha * [x * sum_{k in idx_neg} 1/k - sum_{k in idx_neg} floor(x/k)] <= 1.0001
        
        # The term in brackets is positive for x >= 1 (since sum 1/k > 0 and floor(x/k) <= x/k)
        # Actually, x * sum 1/k - sum floor(x/k) = sum (x/k - floor(x/k)) = sum {x/k} >= 0
        
        # We want to maximize a = -sum f(k) log(k)/k
        # a = -f(1)*log(1)/1 - sum_{k in idx_neg} (-alpha) * log(k)/k
        # a = 0 + alpha * sum_{k in idx_neg} log(k)/k
        
        # So a = alpha * sum_{k in idx_neg} log(k)/k
        
        # To maximize a, we want alpha as large as possible.
        # alpha is constrained by: alpha <= 1.0001 / max_{x in [1,10K]} [x * sum 1/k - sum floor(x/k)]
        
        # Let's compute the maximum of the bracket term.
        # For efficiency, we can compute it for a few x values and estimate.
        
        # Actually, let's just compute it for all x in [1, 10K] but efficiently.
        # g[x] = x * sum_{k in idx_neg} 1/k - sum_{k in idx_neg} floor(x/k)
        
        # sum_{k in idx_neg} 1/k is a constant, call it C1
        C1 = np.sum(1.0 / idx_neg)
        
        # sum_{k in idx_neg} floor(x/k) can be computed by accumulating
        # h[x] = sum_{k in idx_neg} floor(x/k)
        
        # h is a step function. h[x] = sum_{k in idx_neg} floor(x/k)
        # We can compute h for all x in [1, 10K] by:
        # For each k in idx_neg, add 1 to h[x] for x in [k, 2k-1], 2 to h[x] for x in [2k, 3k-1], etc.
        # This is O(K * log K) which might be too slow for K=200000.
        
        # Alternative: compute h[x] directly using the fact that h is the Dirichlet convolution
        # of the indicator of idx_neg with the constant 1 function.
        
        # Let's use a different approach: compute the bracket term for a few sample x values
        # and assume the maximum is at x = 10K or some other specific value.
        
        # Actually, the term x*C1 - sum floor(x/k) is increasing in x?
        # Let's check: d/dx [x*C1 - sum floor(x/k)] = C1 - sum (1/k if x/k not integer else 0)
        # This is not necessarily positive.
        
        # Let's just compute it for all x in [1, 10K] using a vectorized approach.
        # h[x] = sum_{k in idx_neg} floor(x/k)
        # We can compute this by: for each k in idx_neg, h[k::k] += 1, h[2k::2k] += 1, etc.
        # But this is still O(K log K).
        
        # For K=200000, K log K ~ 200000 * 12 = 2.4M operations, which is fine.
        
        h = np.zeros(N + 1, dtype=np.int64)
        for k in idx_neg:
            for j in range(1, N // k + 1):
                h[j*k] += 1
        
        if time.time() - start_time > time_limit - safety_margin:
            return get_fallback()
        
        # Compute the bracket term for all x in [1, N]
        x_vals = np.arange(1, N + 1, dtype=np.float64)
        bracket = x_vals * C1 - h[1:N+1].astype(np.float64)
        
        max_bracket = np.max(bracket)
        
        if max_bracket <= 0:
            continue
        
        alpha = 1.0001 / max_bracket
        
        # Check if alpha is within the clipping range
        # f(k) = -alpha for k in idx_neg, so we need -alpha >= -10, i.e., alpha <= 10
        # f(1) = alpha * C1, we need f(1) <= 10, i.e., alpha * C1 <= 10
        if alpha > 10.0:
            alpha = 10.0
        if alpha * C1 > 10.0:
            alpha = 10.0 / C1
        
        if alpha <= 0:
            continue
        
        # Compute the objective
        a = alpha * np.sum(np.log(idx_neg) / idx_neg)
        
        # Verify M(x) <= 1.0001
        M = alpha * bracket
        if np.max(M) > 1.0001:
            continue
        
        if a > best_a:
            best_a = a
            best_K = K
            best_domain = np.concatenate([np.array([1], dtype=np.int64), idx_neg])
            best_values = np.concatenate([np.array([alpha * C1], dtype=np.float64), 
                                           -alpha * np.ones(len(idx_neg), dtype=np.float64)])
    
    if best_domain is None:
        return get_fallback()
    
    return best_domain, best_values

# EVOLVE-BLOCK-END
