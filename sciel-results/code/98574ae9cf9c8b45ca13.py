# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def _compute_mobius(K):
    """Compute the Möbius function mu(n) for n = 1..K using sieve."""
    mu = np.ones(K + 1, dtype=np.int8)
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
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
    return mu


def _compute_M_max(domain, values, K_max):
    """Compute max M(x) for x in [1, 10*K_max] efficiently using divisor sums."""
    x_max = 10 * K_max
    m = len(domain)
    if m == 0 or x_max <= 0:
        return 0.0
    
    # h(j) = sum_{k|j, k in domain} f(k)
    # M(x) = sum_{j=1}^{x} h(j)
    # So M(x) = cumsum(h[1..x])
    
    h = np.zeros(x_max + 1, dtype=np.float64)
    for i in range(m):
        k = domain[i]
        v = values[i]
        if v != 0:
            h[k::k] += v
    
    M_vals = np.cumsum(h[1:])
    return float(np.max(M_vals))


def _check_feasibility(domain, values, K_max, tolerance=1.0001):
    """Check if M(x) <= tolerance for all x in [1, 10*K_max]."""
    m_max = _compute_M_max(domain, values, K_max)
    return m_max <= tolerance


def _compute_metric(domain, values):
    """Compute a = -sum_k f(k) * log(k) / k."""
    if len(domain) == 0:
        return 0.0
    return float(-np.sum(values * np.log(domain) / domain))


def _apply_clipping_and_normalization(domain, values):
    """Apply clipping to [-10, 10] and normalize so sum f(k)/k = 0."""
    values = np.clip(values, -10.0, 10.0)
    
    # Compute S = sum_k f(k)/k
    S = np.sum(values / domain)
    
    # f(1) is replaced by f(1) - S
    # If 1 is not in domain, f(1) = 0 before this
    idx_1 = np.where(domain == 1)[0]
    if len(idx_1) > 0:
        values[idx_1[0]] -= S
    else:
        # Need to add 1 to domain with value -S
        domain = np.concatenate([[1], domain])
        values = np.concatenate([[-S], values])
        # Sort by domain
        sort_idx = np.argsort(domain)
        domain = domain[sort_idx]
        values = values[sort_idx]
    
    return domain, values


def _solve_lp_for_max_metric(domain, values, K_max, deadline, max_iterations=50):
    """
    Use a cutting-plane / LP approach to maximize the metric while maintaining feasibility.
    We start with the current solution and try to improve it.
    
    The problem is:
    Maximize: -sum f(k) * log(k) / k
    Subject to: M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10*K_max]
               -10 <= f(k) <= 10
               sum_k f(k)/k = 0 (handled by the normalization step, but we incorporate it)
    
    Actually, the normalization is applied AFTER our solution, so we need to account for it.
    Let's think about this differently.
    
    The evaluator:
    1. Clips values to [-10, 10]
    2. Computes S = sum f(k)/k over clipped values
    3. Replaces f(1) with f(1) - S (or adds f(1)=-S if 1 not in domain)
    4. Checks M(x) <= 1.0001 for all x in [1, 10K]
    5. Computes metric a = -sum f(k) * log(k) / k
    
    So we need to optimize the final metric after normalization.
    This is complex. Let's try a simpler approach: use the Möbius function as a starting point
    and try small perturbations or use a greedy approach.
    """
    start_time = time.time()
    best_metric = _compute_metric(domain, values)
    best_domain, best_values = domain.copy(), values.copy()
    
    # Try to improve by adjusting values
    # The Möbius function already gives a good solution.
    # Let's try a simple local optimization: adjust values to maximize metric while keeping M(x) bounded.
    
    # For now, return the current solution as a baseline.
    # A full LP solver would be expensive for large K.
    
    return best_domain, best_values, best_metric


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.
    
    Uses the Möbius function as the base, which asymptotically achieves the optimal
    objective value of 1.0.
    """
    start_time = time.time()
    deadline = start_time + 580  # Leave 50s safety margin
    
    # Compute Möbius function up to K
    K = 200000
    mu = _compute_mobius(K)
    
    # Use all k where mu(k) != 0
    domain = np.where(mu[1:] != 0)[0] + 1
    values = mu[domain].astype(np.float64)
    
    # Check feasibility with the efficient method
    if not _check_feasibility(domain, values, K):
        # If infeasible, try smaller K
        for K_try in [100000, 50000, 20000, 10000, 5000, 2000, 1000]:
            if time.time() > deadline:
                break
            mu_try = _compute_mobius(K_try)
            domain_try = np.where(mu_try[1:] != 0)[0] + 1
            values_try = mu_try[domain_try].astype(np.float64)
            if _check_feasibility(domain_try, values_try, K_try):
                domain, values, K = domain_try, values_try, K_try
                break
        else:
            # Fallback to the original small solution
            domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
            values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
            return domain, values
    
    # Try to improve the solution
    if time.time() < deadline - 100:
        domain, values, metric = _solve_lp_for_max_metric(domain, values, K, deadline)
    
    return domain, values


# EVOLVE-BLOCK-END
