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

    # Strategy: Use the Mobius function mu(k) for k up to K.
    # This maximizes the objective a = -sum f(k) log(k) / k asymptotically.
    # We need to ensure M(x) = sum_{k<=K} mu(k) floor(x/k) <= 1.0001 for x in [1, 10K].
    # Since M(x) is an integer, this is equivalent to M(x) <= 1.
    
    # Try the maximum K first.
    K = 200000
    
    # If time is tight, we might skip the check for smaller Ks, but K=200000 check is fast.
    mu = _compute_mobius(K)
    
    # Construct domain and values
    # We include all k where mu(k) != 0 to maximize the sum.
    # Actually, including mu(k)=0 doesn't change M(x) or the objective, but increases m.
    # The problem says "domain holds distinct positive integers".
    # "values holds the finite reals f(k)".
    # It doesn't strictly forbid f(k)=0, but it's inefficient.
    # However, the evaluator sums over k in domain.
    # Let's stick to non-zero mu(k) to keep m smaller, which is safer for any hidden overhead.
    
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1
    values = mu[domain].astype(np.float64)
    
    # Check feasibility
    # M(x) = sum_{k in domain} mu(k) floor(x/k)
    # We compute M(x) for x in 1..10K
    
    max_x = 10 * K
    M = np.zeros(max_x + 1, dtype=np.int32)
    
    # Optimization: Iterate over domain
    # For each k, add mu[k] to M[x] for x in [j*k, (j+1)*k - 1]
    
    # Check time
    if time.time() > deadline - 5.0:
        # Not enough time to check, fallback to seed
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    for k in domain:
        val = mu[k]
        limit = max_x // k
        for j in range(1, limit + 1):
            start = j * k
            end = (j + 1) * k
            if end > max_x + 1:
                end = max_x + 1
            M[start:end] += val
            
    # Check constraint: M(x) <= 1.0001
    # Since M is integer, M(x) <= 1
    if np.any(M[1:] > 1):
        # Infeasible with K=200000.
        # Try smaller K?
        # The error term is related to Mertens function.
        # Let's try K=100000.
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
                if end > max_x + 1:
                    end = max_x + 1
                M[start:end] += val
        
        if np.any(M[1:] > 1):
            # Still infeasible? Fallback to seed.
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values

    return domain, values


# EVOLVE-BLOCK-END
