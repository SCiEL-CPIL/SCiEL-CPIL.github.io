# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """
    Compute the Mobius function mu(n) for n = 1..K using a linear sieve.
    Returns an array mu of shape (K+1,) where mu[0] is unused.
    """
    mu = np.zeros(K + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_composite = np.zeros(K + 1, dtype=bool)
    
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
    return mu


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = start_time + 600.0 - 10.0  # 10s safety margin

    # We want to approximate the infinite Mobius function.
    # The objective is a = -sum f(k) log(k) / k.
    # The constraint is M(x) = sum f(k) floor(x/k) <= 1.0001 for x in [1, 10K].
    # Ideally M(x) = 1 for all x, which implies f = mu.
    # Then a = -sum mu(k) log(k) / k.
    # As K -> infinity, this sum converges to 1 (since sum mu(n)/n log n is related to log(pi/2) or similar? 
    # Actually, sum_{n=1}^N mu(n)/n = 1/ζ(1) = 0.
    # The sum -sum mu(n) log(n)/n converges to 1?
    # Let's check: sum mu(n)/n log n.
    # It is known that sum_{n<=x} mu(n) = o(x).
    # Partial summation: sum_{n<=x} mu(n) log n / n = -∫ (1/t) d(M(t)) ...
    # Anyway, the task says "optimum 1.0".
    # So we should use as many terms as possible.
    
    # However, we must check feasibility.
    # If we use mu(k) for k <= K, then M(x) = sum_{k<=K} mu(k) floor(x/k).
    # For x <= K, this is exactly 1 (since sum_{d|x} mu(d) = 1 if x=1 else 0? No.
    # Identity: sum_{d|n} mu(d) = 1 if n=1, 0 if n>1.
    # M(x) = sum_{k=1}^x floor(x/k) mu(k) = sum_{k=1}^x mu(k) sum_{j=1}^{floor(x/k)} 1
    # = sum_{j=1}^x sum_{k|j, k<=x} mu(k) = sum_{j=1}^x 1 = x.
    # Wait. The identity is sum_{d|n} mu(d) = [n=1].
    # So sum_{k=1}^x mu(k) floor(x/k) = sum_{k=1}^x mu(k) sum_{j: k|j, j<=x} 1 = sum_{j=1}^x sum_{k|j} mu(k) = sum_{j=1}^x [j=1] = 1.
    # So M(x) = 1 for all x >= 1 if we sum over all k.
    # But we only sum over k <= K.
    # For x <= K, M(x) = 1.
    # For x > K, M(x) = sum_{k<=K} mu(k) floor(x/k).
    # We need M(x) <= 1.0001 for x up to 10K.
    # The error term is sum_{k>K} mu(k) floor(x/k).
    # This error is related to the Mertens function M(x) = sum_{n<=x} mu(n).
    # Specifically, sum_{k=1}^x mu(k) floor(x/k) = 1.
    # So sum_{k>K} mu(k) floor(x/k) = 1 - M_K(x) where M_K(x) = sum_{k<=K} mu(k) floor(x/k).
    # We require M_K(x) <= 1.0001.
    # Since M_K(x) = 1 - Error, we require 1 - Error <= 1.0001 => Error >= -0.0001.
    # Also M_K(x) is an integer? No, mu(k) are integers, floor is integer. So M_K(x) is integer.
    # So M_K(x) <= 1 is the condition.
    # Actually, the condition is M(x) <= 1.0001. Since M(x) is integer, M(x) <= 1.
    # So we need the partial sum to be <= 1.
    
    # Let's try K = 200000.
    # Computing mu up to 200000 is fast.
    # Checking M(x) for x in [1, 10K] might be slow if done naively (10K * K).
    # 10 * 200000 * 200000 = 4 * 10^11. Too slow in Python.
    # We need a faster check or a smaller K.
    # The cost is m * 10K. If m = K, cost is 10 K^2.
    # For K = 200000, 10 * 4*10^10 = 4*10^11. Too slow.
    # We must use a smaller K or a smarter check.
    
    # Let's estimate the time for the check.
    # We can compute M(x) for all x in O(K log K) using a sieve-like accumulation?
    # M(x) = sum_{k<=K} mu(k) floor(x/k).
    # For each k, add mu(k) to M(x) for x in [k, 2k-1], 2mu(k) for x in [2k, 3k-1], etc.
    # This is O(sum_{k<=K} K/k) = O(K log K).
    # K=200000, K log K ~ 200000 * 12 = 2.4 * 10^6. Very fast.
    
    # So we can use K = 200000.
    
    K = 200000
    mu = _compute_mobius(K)
    
    # Construct domain and values
    # domain: k where mu(k) != 0
    # values: mu(k)
    
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1
    values = mu[domain].astype(np.float64)
    
    # Check feasibility
    # Compute M(x) for x in 1..10K
    # M(x) = sum_{k<=K} mu(k) floor(x/k)
    
    # Efficient computation:
    # Initialize M = np.zeros(10K + 1)
    # For each k in domain:
    #   val = mu[k]
    #   for j in 1..floor(10K/k):
    #     M[j*k : (j+1)*k] += val
    
    # This loop is O(K log K).
    
    max_x = 10 * K
    M = np.zeros(max_x + 1, dtype=np.int32)
    
    # We only need to check if M(x) > 1.
    # Note: M(1) = 1. M(x) for x>1 should be 0 if full sum.
    # With partial sum, M(x) can be non-zero.
    
    for k in domain:
        val = mu[k]
        # Add val * floor(x/k) to M(x)
        # floor(x/k) is j for x in [j*k, (j+1)*k - 1]
        # Range of x is 1..max_x
        # j ranges from 1 to max_x // k
        # We can use slicing
        # But slicing with step is not directly additive for this pattern.
        # We can do:
        # for j in range(1, max_x // k + 1):
        #   M[j*k : (j+1)*k] += val
        # This is O(max_x/k) per k. Sum is O(K log K).
        
        # Optimization: use numpy broadcasting or direct loop?
        # Direct loop in Python might be slow if K is large and many iterations.
        # 2.4 * 10^6 iterations is fine.
        
        limit = max_x // k
        for j in range(1, limit + 1):
            start = j * k
            end = (j + 1) * k
            if end > max_x:
                end = max_x + 1
            M[start:end] += val
            
    # Check constraint
    # M(x) <= 1.0001
    # Since M is integer, M(x) <= 1.
    
    if np.any(M[1:] > 1):
        # Infeasible. Try smaller K?
        # Or just return a feasible subset?
        # Let's find the largest prefix that is feasible?
        # Or just reduce K.
        
        # Let's try K = 100000
        K = 100000
        mu = _compute_mobius(K)
        mask = mu[1:] != 0
        domain = np.where(mask)[0] + 1
        values = mu[domain].astype(np.float64)
        
        max_x = 10 * K
        M = np.zeros(max_x + 1, dtype=np.int32)
        
        for k in domain:
            val = mu[k]
            limit = max_x // k
            for j in range(1, limit + 1):
                start = j * k
                end = (j + 1) * k
                if end > max_x:
                    end = max_x + 1
                M[start:end] += val
        
        if np.any(M[1:] > 1):
            # Still infeasible?
            # The error term is sum_{k>K} mu(k) floor(x/k).
            # For x up to 10K, this error is bounded by M(10K) roughly.
            # M(n) is small on average.
            # Maybe K=100000 is safe.
            # If not, fall back to the seed.
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values

    return domain, values


# EVOLVE-BLOCK-END
