# EVOLVE-BLOCK-START
import numpy as np


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Strategy: Dirichlet convolution inversion using the exact Mobius function.
    We compute mu(k) for all k <= K and include all square-free integers in the domain.
    The values are set to mu(k). Then we adjust f(1) so that sum_k f(k)/k = 0.
    This ensures M(x) = sum_k f(k) floor(x/k) is close to 1 (the ideal value) for all x,
    and the objective a = -sum_k f(k) log(k)/k approaches 1 as K grows.
    
    Since M(x) must be <= 1.0001, we need to be careful. With exact mu, 
    M(x) = 1 for all x >= 1 if we include all k <= x. But since we only go up to K,
    for x > K, M(x) = sum_{k<=K} mu(k) floor(x/k).
    
    Actually, the identity sum_{d|n} mu(d) = 1 if n=1 else 0 implies that
    sum_{k<=x} mu(k) floor(x/k) = 1 for all x >= 1.
    
    If we use all k <= K, then for x <= K, M(x) = 1 exactly.
    For x > K, M(x) = sum_{k<=K} mu(k) floor(x/k), which may deviate from 1.
    
    The constraint is M(x) <= 1.0001 for all x in [1, 10K].
    
    Let's use a moderate K to ensure feasibility. We'll compute mu up to K,
    set values to mu(k) for square-free k, and adjust f(1).
    """
    # Choose K: large enough to get a good approximation, small enough for speed
    K = 50000
    
    # Compute Mobius function up to K using sieve
    mu = np.ones(K + 1, dtype=np.int64)
    primes = []
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    
    # Sieve of Eratosthenes to find primes
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            for j in range(i * i, K + 1, i):
                is_prime[j] = False
    
    # Compute mu using the multiplicative property
    mu = np.ones(K + 1, dtype=np.float64)
    for p in primes:
        for j in range(p, K + 1, p):
            mu[j] *= -1
        p2 = p * p
        for j in range(p2, K + 1, p2):
            mu[j] = 0.0
    
    # Get square-free indices (where mu != 0)
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1
    values = mu[domain]
    
    # Ensure f(1) exists in the domain
    if domain[0] != 1:
        # This shouldn't happen since mu(1) = 1
        domain = np.insert(domain, 0, 1)
        values = np.insert(values, 0, 1.0)
    
    # Compute S = sum_k f(k)/k over the current values
    S = np.sum(values / domain.astype(np.float64))
    
    # Adjust f(1) so that sum_k f(k)/k = 0
    # f(1) is at index 0 (since domain[0] = 1)
    values[0] -= S
    
    # Clip values to [-10, 10] as per the evaluation protocol
    values = np.clip(values, -10.0, 10.0)
    
    # Re-compute S after clipping and re-adjust f(1)
    S = np.sum(values / domain.astype(np.float64))
    values[0] -= S
    
    # Clip again
    values = np.clip(values, -10.0, 10.0)
    
    # Final adjustment: since clipping might have changed things, 
    # we need to ensure the constraint sum f(k)/k = 0 is met.
    # But the evaluator does: f(1) = f(1) - S where S = sum f(k)/k over clipped values.
    # So we should provide values BEFORE the evaluator's adjustment.
    # The evaluator will handle the f(1) adjustment itself.
    
    # Let me re-read the task:
    # "values are clipped to [-10, 10]. Then f(1) is replaced by f(1) - S with S = sum_k f(k)/k"
    # So the evaluator does the clipping and f(1) adjustment.
    # We just need to provide the raw values.
    
    # So we should return the mu values directly, without our own adjustment.
    # The evaluator will clip and adjust f(1).
    
    # Reset values to mu
    values = mu[domain]
    
    # Return domain and values
    return domain, values


# EVOLVE-BLOCK-END
