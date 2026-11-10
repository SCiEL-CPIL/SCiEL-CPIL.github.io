# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mobius(N):
    """Compute Moebius function mu(k) for k = 1..N using linear sieve."""
    mu = np.zeros(N + 1, dtype=np.int8)
    mu[1] = 1
    is_prime = np.ones(N + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, N + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
        for p in primes:
            if i * p > N:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]
    return mu


def _compute_F_array(mu, K, max_x):
    """
    Compute F(x) = sum_{k=1}^K mu(k)*floor(x/k) for x in [1, max_x].
    Uses the identity F(x) = sum_{n=1}^x g(n) where g(n) = sum_{k|n, k<=K} mu(k).
    """
    g = np.zeros(max_x + 1, dtype=np.float64)
    for k in range(1, K + 1):
        v = mu[k]
        if v != 0:
            g[k::k] += v
    F = np.cumsum(g)
    return F[1:max_x + 1]  # F(x) for x = 1..max_x


def _compute_S_mu(mu, K):
    """Compute S = sum_{k=1}^K mu(k)/k."""
    mask = mu[1:K + 1] != 0
    if not np.any(mask):
        return 0.0
    ks = np.where(mask)[0] + 1
    mu_vals = mu[1:K + 1][mask].astype(np.float64)
    return np.sum(mu_vals / ks.astype(np.float64))


def _compute_obj_mu(mu, K):
    """Compute sum_{k=1}^K mu(k)*log(k)/k."""
    mask = mu[1:K + 1] != 0
    if not np.any(mask):
        return 0.0
    ks = np.where(mask)[0] + 1
    mu_vals = mu[1:K + 1][mask].astype(np.float64)
    log_k = np.log(ks.astype(np.float64))
    return np.sum(mu_vals * log_k / ks.astype(np.float64))


def _find_optimal_scaling(mu, K, max_x, S_mu, obj_mu):
    """
    Find the maximum scaling factor c >= 0 such that:
    M(x) = c * F(x) - S_actual * x <= 1.0001 for all x in [1, max_x]
    
    After scaling, values = c * mu(k) for k=1..K.
    S_actual = c * S_mu (since sum c*mu(k)/k = c*S_mu)
    So M(x) = c * F(x) - c * S_mu * x = c * (F(x) - S_mu * x)
    
    We need c * max_x [F(x) - S_mu * x] <= 1.0001
    So c_max = 1.0001 / max(0, max_x [F(x) - S_mu * x])
    
    The objective is a = -c * obj_mu (since log(1)=0, the f(1) adjustment doesn't affect it)
    So we want to maximize c * (-obj_mu) = c * |obj_mu| (since obj_mu < 0 for large K)
    """
    F = _compute_F_array(mu, K, max_x)
    xs = np.arange(1, max_x + 1, dtype=np.float64)
    
    # h(x) = F(x) - S_mu * x
    h = F - S_mu * xs
    
    max_h = np.max(h)
    
    if max_h <= 0:
        # No constraint, c can be up to 10
        c_max = 10.0
    else:
        c_max = 1.0001 / max_h
        c_max = min(c_max, 10.0)
    
    return c_max, max_h


def _local_optimize_c(K, c_start, mu, max_x, S_mu, obj_mu, deadline, tol=1e-10, max_iter=50):
    """
    Local optimization of c to maximize objective while maintaining feasibility.
    Uses bisection on the constraint boundary for precision.
    """
    best_c = c_start
    best_a = -np.inf
    
    # Compute the constraint function h(x) = F(x) - S_mu * x
    # M(x) = c * h(x) <= 1.0001
    # So c_max = 1.0001 / max(h(x)) if max(h) > 0
    
    F = _compute_F_array(mu, K, max_x)
    xs = np.arange(1, max_x + 1, dtype=np.float64)
    h = F - S_mu * xs
    max_h = np.max(h)
    
    if max_h <= 0:
        # No constraint from M(x), just bound by clipping
        # Try c values near 10
        for c_try in [10.0, 9.9, 9.95, 9.99]:
            if time.time() > deadline - 5:
                break
            a_try, feas_try = _evaluate_candidate(mu, K, c_try, S_mu, obj_mu, max_x)
            if feas_try and a_try is not None and a_try > best_a:
                best_a = a_try
                best_c = c_try
        return best_c, best_a
    
    # Bisection to find precise c_max
    c_lo = 0.0
    c_hi = 1.0001 / max_h
    
    # Verify c_hi is feasible
    a_hi, feas_hi = _evaluate_candidate(mu, K, c_hi, S_mu, obj_mu, max_x)
    if feas_hi and a_hi is not None:
        best_a = a_hi
        best_c = c_hi
    
    # Bisection: find largest c such that M(x) <= 1.0001
    for _ in range(max_iter):
        if time.time() > deadline - 10:
            break
        c_mid = 0.5 * (c_lo + c_hi)
        if abs(c_hi - c_lo) < tol:
            break
        
        a_mid, feas_mid = _evaluate_candidate(mu, K, c_mid, S_mu, obj_mu, max_x)
        if feas_mid and a_mid is not None:
            best_a = max(best_a, a_mid)
            best_c = c_mid
            c_lo = c_mid
        else:
            c_hi = c_mid
    
    # Try a few values near the boundary for precision
    for c_try in [best_c, best_c * 0.999, best_c * 0.9999, best_c * 1.0001]:
        if time.time() > deadline - 5:
            break
        if c_try <= 0 or c_try > 10:
            continue
        a_try, feas_try = _evaluate_candidate(mu, K, c_try, S_mu, obj_mu, max_x)
        if feas_try and a_try is not None and a_try > best_a:
            best_a = a_try
            best_c = c_try
    
    return best_c, best_a


def _evaluate_candidate(mu, K, c, S_mu, obj_mu, max_x):
    """
    Evaluate a candidate (K, c) and return the objective value and feasibility.
    """
    # Construct values
    mu_K = mu[1:K + 1].astype(np.float64)
    values = c * mu_K
    
    # Clip to [-10, 10]
    values = np.clip(values, -10.0, 10.0)
    
    # Compute actual S after clipping
    ks = np.arange(1, K + 1, dtype=np.float64)
    S_actual = np.sum(values / ks)
    
    # Check feasibility: M(x) = c*F(x) - S_actual*x <= 1.0001
    # But wait, after clipping, the values are no longer exactly c*mu(k)
    # So we need to recompute M(x) with the clipped values
    # M(x) = sum_{k=1}^K g(k) * floor(x/k) where g(1) = values[0] - S_actual, g(k) = values[k-1] for k>=2
    # M(x) = sum_{k=1}^K values[k-1]*floor(x/k) - S_actual*x
    
    # Compute sum_{k=1}^K values[k-1]*floor(x/k) for all x in [1, max_x]
    g = np.zeros(max_x + 1, dtype=np.float64)
    for k in range(1, K + 1):
        v = values[k - 1]
        if v != 0:
            g[k::k] += v
    M_direct = np.cumsum(g)
    
    # M(x) = M_direct(x) - S_actual * x
    xs = np.arange(1, max_x + 1, dtype=np.float64)
    M = M_direct[1:max_x + 1] - S_actual * xs
    
    if np.max(M) > 1.0001:
        return None, False
    
    # Compute objective: a = -sum_{k=1}^K g(k) * log(k) / k
    # g(1) = values[0] - S_actual, g(k) = values[k-1] for k >= 2
    # a = -(values[0] - S_actual)*log(1)/1 - sum_{k=2}^K values[k-1]*log(k)/k
    # = 0 - sum_{k=2}^K values[k-1]*log(k)/k
    # = -sum_{k=1}^K values[k-1]*log(k)/k (since log(1) = 0)
    
    log_k = np.log(ks)
    a = -np.sum(values * log_k / ks)
    
    return a, True


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = 630.0

    K_max = 200000
    mu_full = _compute_mobius(K_max)

    best_domain = None
    best_values = None
    best_a = -np.inf

    # Try a range of K values
    K_candidates = list(range(50, 500, 10)) + list(range(500, 2000, 25)) + \
                   list(range(2000, 10000, 100)) + list(range(10000, 50000, 250)) + \
                   list(range(50000, 200001, 1000))
    K_candidates = sorted(set(K_candidates))

    results = []
    for K in K_candidates:
        if time.time() > deadline - 120:
            break

        max_x = 10 * K
        S_mu = _compute_S_mu(mu_full, K)
        obj_mu = _compute_obj_mu(mu_full, K)

        # Find maximum feasible c
        c_max, max_h = _find_optimal_scaling(mu_full, K, max_x, S_mu, obj_mu)

        if c_max <= 0:
            continue

        # The objective with optimal c (before clipping)
        a_est = -c_max * obj_mu

        results.append((K, c_max, a_est, S_mu, obj_mu, max_h))

    # Sort by estimated objective descending
    results.sort(key=lambda x: x[2], reverse=True)

    # Verify top candidates with local optimization
    for K, c, a_est, S_mu, obj_mu, max_h in results[:50]:
        if time.time() > deadline - 60:
            break

        max_x = 10 * K

        # Use local optimization to find precise best c
        c_opt, a_opt = _local_optimize_c(K, c, mu_full, max_x, S_mu, obj_mu, deadline)
        
        if a_opt > best_a:
            best_a = a_opt
            domain = np.arange(1, K + 1, dtype=np.int64)
            values = (c_opt * mu_full[1:K + 1].astype(np.float64))
            values = np.clip(values, -10.0, 10.0)
            best_domain = domain.copy()
            best_values = values.copy()

        # If we're close to the theoretical max, try to polish with more K values
        if best_a > 0.99:
            # Try nearby K values for potential improvement
            for K_try in [K - 50, K - 25, K + 25, K + 50]:
                if time.time() > deadline - 30:
                    break
                if K_try < 10 or K_try > K_max:
                    continue
                max_x_try = 10 * K_try
                S_mu_try = _compute_S_mu(mu_full, K_try)
                obj_mu_try = _compute_obj_mu(mu_full, K_try)
                c_try, _ = _find_optimal_scaling(mu_full, K_try, max_x_try, S_mu_try, obj_mu_try)
                if c_try <= 0:
                    continue
                c_opt_try, a_opt_try = _local_optimize_c(K_try, c_try, mu_full, max_x_try, S_mu_try, obj_mu_try, deadline)
                if a_opt_try > best_a:
                    best_a = a_opt_try
                    domain = np.arange(1, K_try + 1, dtype=np.int64)
                    values = (c_opt_try * mu_full[1:K_try + 1].astype(np.float64))
                    values = np.clip(values, -10.0, 10.0)
                    best_domain = domain.copy()
                    best_values = values.copy()
            break

    # Fallback
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    # Final safety check: verify the returned values are valid
    # Clip values to ensure they're in valid range
    values = np.clip(values, -10.0, 10.0)
    
    return domain, values


# EVOLVE-BLOCK-END
