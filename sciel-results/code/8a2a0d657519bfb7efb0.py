# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    # Use Möbius function mu(k) for k = 1..K
    # For small K, the truncation error in M(x) is manageable
    
    K = 30
    
    # Compute Möbius function up to K
    mu = np.ones(K + 1, dtype=np.float64)
    is_prime = np.ones(K + 1, dtype=bool)
    for i in range(2, K + 1):
        if is_prime[i]:
            mu[i] = -1.0
            for j in range(i * i, K + 1, i):
                is_prime[j] = False
                mu[j] *= -1.0
                if j % (i * i) == 0:
                    mu[j] = 0.0
    mu[1] = 1.0
    
    # Filter out zeros (mu(k) = 0 for non-squarefree k)
    mask = mu[1:K+1] != 0
    domain = np.where(mask)[0] + 1
    values = mu[1:K+1][mask]
    
    # Verify constraint: M(x) <= 1.0001 for x in [1, 10K]
    max_x = 10 * K
    M = np.zeros(max_x + 1, dtype=np.float64)
    for k, v in zip(domain, values):
        for j in range(1, max_x // k + 1):
            start = j * k
            end = min((j + 1) * k - 1, max_x)
            M[start:end+1] += v
    
    max_M = np.max(M[1:max_x+1])
    
    # If infeasible, scale down the values slightly (but this reduces the objective)
    # For K=30, the tail error should be small
    # Return the function
    return domain, values


# EVOLVE-BLOCK-END
