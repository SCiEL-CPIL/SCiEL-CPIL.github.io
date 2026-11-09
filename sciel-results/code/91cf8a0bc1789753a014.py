# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy: Dirichlet convolution inversion using the exact Mobius function mu(k).
    We compute mu(k) for all k <= K and retain only square-free indices (mu(k) != 0).
    The values are set to mu(k). This approximates the infinite identity where
    sum_k mu(k)/k = 0 and M(x) = sum_k mu(k) * floor(x/k) = 1 for x >= 1.

    After the evaluator's clipping and the f(1) adjustment, the resulting function
    should have M(x) <= 1.0001 for all x in [1, 10K] and a high objective value
    a = -sum_k f(k) * log(k) / k.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    # Maximum key K as per task contract
    K = 200000

    # Compute the Mobius function mu(k) for all k in [1, K]
    # mu(n) = 0 if n has a squared prime factor
    # mu(n) = (-1)^r if n is a product of r distinct primes
    # mu(1) = 1

    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0

    # Sieve of Eratosthenes to compute mu
    # Initialize mu as 1 for all, then for each prime p, multiply mu by -1 for multiples of p,
    # and set mu to 0 for multiples of p^2.

    # First, mark primes using a simple sieve
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False

    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            if i * i <= K:
                is_prime[i * i::i] = False

    # Now compute mu using the primes
    # Start with mu = 1 for all
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0

    for p in primes:
        # For multiples of p, flip the sign of mu
        mu[p::p] *= -1
        # For multiples of p^2, set mu to 0
        p_sq = p * p
        if p_sq <= K:
            mu[p_sq::p_sq] = 0

    # domain: indices where mu(k) != 0 (square-free numbers)
    mask = mu != 0
    domain = np.nonzero(mask)[0]
    # Filter out 0 (we want positive integers)
    domain = domain[domain > 0]
    values = mu[domain].astype(np.float64)

    return domain, values


# EVOLVE-BLOCK-END
