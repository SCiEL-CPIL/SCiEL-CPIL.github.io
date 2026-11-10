# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(K):
    """Compute the Möbius function μ(n) for n = 1..K using a linear sieve."""
    mu = np.ones(K + 1, dtype=np.float64)
    is_prime = np.ones(K + 1, dtype=bool)
    primes = []
    mu[0] = 0
    is_prime[0] = False
    is_prime[1] = False

    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1.0
        for p in primes:
            if i * p > K:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0.0
                break
            else:
                mu[i * p] = -mu[i]
    return mu


def _compute_M_max(domain, values, K):
    """Compute max M(x) for x in [1, 10K] and return (max_val, argmax_x)."""
    max_x = 10 * K
    g = np.zeros(max_x + 1, dtype=np.float64)
    for k, v in zip(domain, values):
        if v != 0:
            g[k::k] += v
    G = np.cumsum(g[1:])  # G[i] = M(i+1)
    idx = np.argmax(G)
    return G[idx], idx + 1


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Strategy: Use the Möbius function μ(n) as the base. The identity Σ_{d|n} μ(d) = 0
    ensures M(x) = 1 for x < K (approximately), so the constraint M(x) ≤ 1 is naturally
    satisfied for x up to ~2K. For x > 2K, M(x) may exceed 1 slightly. We compute the
    exact maximum and if it exceeds 1.0001, we apply a minimal uniform scaling factor.
    The objective a = -Σ f(k)·log(k)/k is maximized when f = μ, so we want minimal
    deviation from μ.
    """
    start_time = time.time()
    deadline = 620  # Safety margin before 630s limit

    # Try K values from large to small
    for K in [200000, 150000, 100000, 75000, 50000, 30000, 20000, 10000, 5000, 1000, 100]:
        if time.time() - start_time > deadline:
            break

        mu = _compute_mobius(K)
        nonzero_mask = mu[1:] != 0
        domain = np.where(nonzero_mask)[0] + 1
        values = mu[1:][nonzero_mask].copy()

        # Check if raw μ is feasible
        max_M, _ = _compute_M_max(domain, values, K)
        
        if max_M <= 1.0001:
            # Raw μ is feasible — return it directly (optimal)
            return domain, values
        else:
            # Need to scale down. Find the minimal c such that c*μ is feasible.
            # Since M(x) is linear in c, if max M(x) = M_max, we need c * M_max <= 1.0001
            # So c <= 1.0001 / M_max
            c = 1.0001 / max_M
            values_scaled = values * c
            # Verify feasibility after scaling
            max_M2, _ = _compute_M_max(domain, values_scaled, K)
            if max_M2 <= 1.0001:
                return domain, values_scaled
            # If still not feasible (shouldn't happen due to linearity), try smaller
            c = 0.9999 / max_M
            values_scaled = values * c
            max_M3, _ = _compute_M_max(domain, values_scaled, K)
            if max_M3 <= 1.0001:
                return domain, values_scaled

    # Fallback to seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values


# EVOLVE-BLOCK-END
