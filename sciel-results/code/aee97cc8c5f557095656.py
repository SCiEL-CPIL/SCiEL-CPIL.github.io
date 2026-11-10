# EVOLVE-BLOCK-START
import numpy as np
import time

def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    The function f is scored by a fixed evaluator (see evaluator.py) that:
      * subtracts a constant from f(1) so that sum_k f(k)/k == 0,
      * requires the step function M(x) = sum_k f(k) * floor(x/k) to stay <= 1 for every
        real x in [1, 10*max_key] (otherwise the construction is infeasible), and
      * returns the objective a = -sum_k f(k) * log(k) / k (to be maximized; optimum 1.0).
    The optimum is approached by the Mobius function mu(k), for which the classic identity
    sum_{k<=x} mu(k) floor(x/k) = 1 holds and -sum mu(k) log(k)/k -> 1.

    Representation:
        domain : np.ndarray, shape (m,), dtype int   -> positive integers in the domain of f
        values : np.ndarray, shape (m,), dtype float -> the corresponding real values f(k)

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = 600.0  # 600 seconds limit, keep margin
    
    # Maximum possible K
    K_max = 200000
    
    # 1. Compute Mobius function up to K_max
    mu = np.zeros(K_max + 1, dtype=np.int8)
    mu[1] = 1
    primes = []
    is_prime = np.ones(K_max + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    # Linear Sieve for Mobius
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
                
    # 2. Determine the largest K such that the constraint holds
    # We start with K_max and check. If it fails, we might try smaller K, 
    # but larger K is generally better for the objective and likely better for feasibility 
    # (smaller truncation error). 
    # However, the error term M(10K) - M(K) can fluctuate.
    # Let's just check K_max.
    
    K = K_max
    valid = False
    
    # Check feasibility for K
    # M(x) = sum_{k=1}^K mu(k) floor(x/k)
    # We compute S(n) = sum_{k|n, k<=K} mu(k) for n=1..10K
    # Then M(x) = sum_{n=1}^x S(n)
    
    max_x = 10 * K
    S = np.zeros(max_x + 1, dtype=np.int32)
    
    # Efficiently compute S(n)
    # Iterate k from 1 to K
    # For each k, add mu[k] to S[j] for all j that are multiples of k
    # This is O(K log K)
    
    # To respect time limit, we do this in a loop with time checks
    for k in range(1, K + 1):
        if time.time() > deadline - 60:
            # If running out of time, fallback to a smaller K or current best
            # But this loop is fast enough for K=200k
            break
        mu_k = mu[k]
        if mu_k != 0:
            # Add mu_k to multiples of k
            # S[k::k] += mu_k
            # This is a slice operation, very fast in numpy
            S[k::k] += mu_k
            
    # Compute prefix sums to get M(x)
    M = np.cumsum(S)
    
    # Check constraint: M(x) <= 1.0001 for all x in [1, 10K]
    # M is 1-indexed effectively if we consider S[1..max_x]
    # S[0] is 0.
    # M[x] is sum_{n=1}^x S[n]
    
    max_M = np.max(M[1:max_x+1])
    
    if max_M <= 1.0001:
        valid = True
    else:
        # If K_max is not valid, try to find a valid K.
        # Usually, smaller K has larger error, so this is tricky.
        # But let's try a binary search or just reduce K significantly.
        # For now, if K_max fails, we try a smaller K, e.g., 50000.
        # In practice, K_max should work.
        
        # Let's try a few smaller values
        for K_try in [100000, 50000, 10000]:
            if time.time() > deadline - 30:
                break
            K = K_try
            max_x = 10 * K
            S = np.zeros(max_x + 1, dtype=np.int32)
            for k in range(1, K + 1):
                mu_k = mu[k]
                if mu_k != 0:
                    S[k::k] += mu_k
            M = np.cumsum(S)
            max_M = np.max(M[1:max_x+1])
            if max_M <= 1.0001:
                valid = True
                break
        
    if not valid:
        # Fallback to the seed program's domain if all else fails
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values

    # Construct domain and values for the valid K
    # Domain: indices where mu[k] != 0
    # Values: mu[k]
    
    mask = mu[1:K+1] != 0
    domain = np.where(mask)[0] + 1
    values = mu[1:K+1][mask].astype(np.float64)
    
    return domain, values

# EVOLVE-BLOCK-END
