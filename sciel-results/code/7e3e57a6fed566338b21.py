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
    K = 200000
    
    # Compute Mobius function mu(k) for k=1..K using a sieve
    # mu(1) = 1
    # mu(n) = 0 if n has a squared prime factor
    # mu(n) = (-1)^k if n is a squarefree integer with k prime factors
    
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    
    # Sieve of Eratosthenes variant for Mobius
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            # Mark multiples of i
            for j in range(i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1
            
    # Handle squared prime factors
    for p in primes:
        p2 = p * p
        if p2 > K:
            break
        for j in range(p2, K + 1, p2):
            mu[j] = 0

    # Filter domain to only include k where mu(k) != 0
    # We include k=1 as well
    mask = mu != 0
    domain = np.where(mask)[0].astype(np.int64)
    values = mu[domain].astype(np.float64)
    
    # Check time just in case, though this is very fast
    elapsed = time.time() - start_time
    if elapsed > 580:
         # Fallback to seed if somehow slow (unlikely)
         domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
         values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
         
    return domain, values

# EVOLVE-BLOCK-END
