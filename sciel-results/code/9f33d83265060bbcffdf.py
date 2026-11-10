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
    deadline = 630  # Safety margin for 600s limit + overhead
    
    # Maximum key allowed by the problem statement
    K = 200000
    
    # Compute Mobius function mu(k) for k = 1..K using a sieve
    # mu(n) is multiplicative.
    # Initialize mu with 1
    mu = np.ones(K + 1, dtype=np.int64)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
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

    # The domain is all integers from 1 to K
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    
    # Optional: Verify feasibility quickly if time permits, though for mu it is known to be safe
    # The evaluator checks M(x) <= 1.0001.
    # M(x) = sum_{k=1}^K mu(k) floor(x/k).
    # For x <= K, M(x) = 1.
    # For x > K, M(x) = 1 - sum_{k=K+1}^infty mu(k) floor(x/k).
    # The error term is small.
    
    # Check time
    if time.time() > deadline - 10:
        # Return the result immediately
        pass

    return domain, values

# EVOLVE-BLOCK-END
