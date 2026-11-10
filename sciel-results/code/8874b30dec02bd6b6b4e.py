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
    deadline = start_time + 500  # Safety margin

    # Compute the Mobius function mu(k) for k = 1..K using a linear sieve.
    K = 200000
    
    # Linear sieve for Mobius function
    mu = [1] * (K + 1)
    is_prime = [True] * (K + 1)
    primes = []
    
    # mu[0] is unused, mu[1] = 1
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
                
    if time.time() > deadline:
        # Fallback: Return minimal valid set
        return np.array([1]), np.array([0.0])

    # Convert to NumPy arrays
    mu_arr = np.array(mu[1:], dtype=np.int8)
    
    # Collect nonzero mu values
    mask = mu_arr != 0
    domain = np.where(mask)[0] + 1
    values = mu_arr[mask].astype(np.float64)
    
    return domain, values


# EVOLVE-BLOCK-END
