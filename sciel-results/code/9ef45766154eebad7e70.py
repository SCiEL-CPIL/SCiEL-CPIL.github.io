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


def _apply_f1_adjustment(domain, values):
    """Apply the f(1) adjustment: clip to [-10,10], then adjust f(1) so sum f(k)/k = 0."""
    values_clipped = np.clip(values, -10, 10)
    
    # Compute S = sum_k f(k)/k over clipped values
    k_float = domain.astype(np.float64)
    S = np.sum(values_clipped / k_float)
    
    # Adjust f(1)
    if len(domain) > 0 and domain[0] == 1:
        # 1 is in domain, adjust it
        values_adjusted = values_clipped.copy()
        values_adjusted[0] = values_adjusted[0] - S
        return domain, values_adjusted
    else:
        # 1 is not in domain, add it with value -S
        values_adjusted = np.concatenate([[ -S ], values_clipped])
        domain_new = np.concatenate([[1], domain])
        return domain_new, values_adjusted


def _compute_M_values(domain, values, K):
    """Compute M(x) for all x in [1, 10K] efficiently."""
    max_x = 10 * K
    g = np.zeros(max_x + 1, dtype=np.float64)
    for k, v in zip(domain, values):
        if v != 0:
            g[k::k] += v
    G = np.cumsum(g[1:])  # G[i] = M(i+1)
    return G


def _is_feasible(domain, values, K):
    """Check if M(x) <= 1.0001 for all x in [1, 10K]."""
    G = _compute_M_values(domain, values, K)
    return np.max(G) <= 1.0001


def _compute_objective(domain, values):
    """Compute the objective a = -sum_k f(k) * log(k) / k."""
    k_float = domain.astype(np.float64)
    v_float = values.astype(np.float64)
    return -np.sum(v_float * np.log(k_float) / k_float)


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    """
    start_time = time.time()
    deadline = 610  # Safety margin before 630s limit

    best_result = None
    best_score = -np.inf

    # Try different K values
    for K in [200000, 100000, 50000, 20000, 10000, 5000, 1000, 100]:
        if time.time() - start_time > deadline:
            break

        mu = _compute_mobius(K)
        nonzero_mask = mu[1:] != 0
        domain = np.where(nonzero_mask)[0] + 1
        values_base = mu[1:][nonzero_mask].copy()

        # Binary search for optimal scaling factor c
        # f(k) = c * mu(k), then apply f(1) adjustment
        # We want to maximize the objective while keeping M(x) <= 1.0001

        # First, check if c=1 is feasible
        values_test = values_base * 1.0
        domain_adj, values_adj = _apply_f1_adjustment(domain, values_test)
        if _is_feasible(domain_adj, values_adj, K):
            # c=1 is feasible, try to increase c
            lo, hi = 1.0, 10.0
            for _ in range(40):
                if time.time() - start_time > deadline:
                    break
                mid = (lo + hi) / 2
                values_test = values_base * mid
                domain_adj, values_adj = _apply_f1_adjustment(domain, values_test)
                if _is_feasible(domain_adj, values_adj, K):
                    lo = mid
                else:
                    hi = mid
            c_opt = lo
        else:
            # c=1 is not feasible, try to decrease c
            lo, hi = 0.0, 1.0
            for _ in range(40):
                if time.time() - start_time > deadline:
                    break
                mid = (lo + hi) / 2
                values_test = values_base * mid
                domain_adj, values_adj = _apply_f1_adjustment(domain, values_test)
                if _is_feasible(domain_adj, values_adj, K):
                    lo = mid
                else:
                    hi = mid
            c_opt = lo

        if c_opt > 0:
            values = values_base * c_opt
            domain_adj, values_adj = _apply_f1_adjustment(domain, values)
            if _is_feasible(domain_adj, values_adj, K):
                score = _compute_objective(domain_adj, values_adj)
                if score > best_score:
                    best_score = score
                    best_result = (domain_adj.copy(), values_adj.copy())

    if best_result is not None:
        return best_result

    # Fallback to seed
    domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    return domain, values


# EVOLVE-BLOCK-END
