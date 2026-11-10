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
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    start_time = time.time()
    deadline = 630.0  # Safety margin from 600s limit
    
    # Maximum allowed key K
    K = 200000
    
    # Compute Mobius function mu(k) for k in [1, K]
    # mu(1) = 1
    # mu(n) = 0 if n has a squared prime factor
    # mu(n) = (-1)^r if n is a product of r distinct primes
    
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    
    # Sieve of Eratosthenes to find primes and compute mu
    for i in range(2, K + 1):
        if is_prime[i]:
            # i is a prime
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            # Handle squared factors
            for j in range(i * i, K + 1, i * i):
                mu[j] = 0
    
    # Create domain and values arrays
    # We include all k in [1, K]
    domain = np.arange(1, K + 1, dtype=np.int64)
    values = mu[1:].astype(np.float64)
    
    # Check time
    if time.time() > deadline:
        # Fallback to a smaller domain if computation took too long (unlikely for K=200000)
        # But we should return something valid.
        # Since the sieve is fast, this shouldn't happen.
        pass
        
    return domain, values

# EVOLVE-BLOCK-END
