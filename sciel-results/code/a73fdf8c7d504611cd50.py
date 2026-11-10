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


def _compute_F_max(mu, K, max_x):
    """
    Compute max_{x in [1, max_x]} F(x) where F(x) = sum_{k=1}^K mu(k)*floor(x/k).
    F(x) = sum_{n=1}^x g(n) where g(n) = sum_{k|n, k<=K} mu(k).
    """
    g = np.zeros(max_x + 1, dtype=np.float64)
    for k in range(1, K + 1):
        v = mu[k]
        if v != 0:
            g[k::k] += v
    F = np.cumsum(g)
    return np.max(F[1:max_x + 1]) if max_x >= 1 else 0.0


def _compute_H(K):
    """Compute H_K = sum_{k=1}^K 1/k."""
    return np.sum(1.0 / np.arange(1, K + 1))


def _compute_S_mu(mu, K):
    """Compute S = sum_{k=1}^K mu(k)/k."""
    mask = mu[1:K + 1] != 0
    ks = np.where(mask)[0] + 1
    if len(ks) == 0:
        return 0.0
    mu_vals = mu[1:K + 1][mask].astype(np.float64)
    return np.sum(mu_vals / ks.astype(np.float64))


def _compute_obj_mu(mu, K):
    """Compute sum_{k=1}^K mu(k)*log(k)/k."""
    mask = mu[1:K + 1] != 0
    ks = np.where(mask)[0] + 1
    if len(ks) == 0:
        return 0.0
    mu_vals = mu[1:K + 1][mask].astype(np.float64)
    log_k = np.log(ks.astype(np.float64))
    return np.sum(mu_vals * log_k / ks.astype(np.float64))


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f
    """
    start_time = time.time()
    deadline = 600.0

    K_max = 200000
    mu_full = _compute_mobius(K_max)

    best_domain = None
    best_values = None
    best_a = -np.inf

    # Try a range of K values
    K_candidates = list(range(50, 500, 10)) + list(range(500, 2000, 50)) + \
                   list(range(2000, 10000, 200)) + list(range(10000, 50000, 1000)) + \
                   list(range(50000, 200001, 5000))
    K_candidates = sorted(set(K_candidates))

    results = []
    for K in K_candidates:
        if time.time() > deadline - 120:
            break

        max_x = 10 * K
        mu_K = mu_full[1:K + 1]

        # Compute max F(x) for x in [1, 10K]
        max_F = _compute_F_max(mu_full, K, max_x)

        # Compute S = sum mu(k)/k
        S = _compute_S_mu(mu_full, K)

        # Compute H_K = sum 1/k
        H_K = _compute_H(K)

        # Compute sum mu(k)*log(k)/k
        obj_mu = _compute_obj_mu(mu_full, K)

        # We set f(k) = c*mu(k) for all k in [1, K]
        # After f(1) adjustment: g(k) = f(k) - S_total where S_total = sum f(k)/k = c*S
        # M(x) = F(x) - c*S*(x-1) where F(x) = sum mu(k)*floor(x/k)
        # Constraint: max_x [F(x) - c*S*(x-1)] <= 1.0001
        # Also: c*mu(k) must be in [-10, 10] before clipping, so c <= 10

        # Objective: a = -c*obj_mu + c*S*H_K = c*(S*H_K - obj_mu)

        # Find optimal c:
        # max_x [F(x) - c*S*(x-1)] <= 1.0001
        # This is max over x of [F(x) - c*S*(x-1)]
        # For a given c, this is a linear function in c for each x
        # The constraint is: for all x, F(x) - c*S*(x-1) <= 1.0001
        # i.e., c*S*(x-1) >= F(x) - 1.0001 for all x
        # If S > 0: c >= (F(x) - 1.0001) / (S*(x-1)) for all x where x > 1
        # If S < 0: c <= (F(x) - 1.0001) / (S*(x-1)) for all x where x > 1 (note S*(x-1) < 0)

        # Let's compute the binding constraint
        # For x = 1: F(1) - 0 = F(1) = mu(1) = 1, so 1 <= 1.0001 is always satisfied
        # For x > 1: F(x) - c*S*(x-1) <= 1.0001

        # Let's find the maximum of (F(x) - 1.0001) / (S*(x-1)) for x > 1
        # But we need to be careful about the sign of S

        # Actually, let's just try a range of c values and check feasibility
        # c must be in [0, 10] (since mu(k) in {-1,0,1}, |c*mu(k)| = c for non-zero)

        # The constraint is: max_x [F(x) - c*S*(x-1)] <= 1.0001
        # Let's define h(c) = max_x [F(x) - c*S*(x-1)]
        # We want h(c) <= 1.0001

        # h(c) is the max of linear functions in c, so it's convex in c
        # We want to find the largest c such that h(c) <= 1.0001 (if S*H_K - obj_mu > 0)
        # or the smallest c (if we want to minimize |c| for negative objective coefficient)

        # Let's compute the objective coefficient
        obj_coeff = S * H_K - obj_mu

        if abs(obj_coeff) < 1e-15:
            continue

        # If obj_coeff > 0, we want to maximize c
        # If obj_coeff < 0, we want to minimize c (but c >= 0)

        # Let's find the feasible range of c
        # h(c) = max_x [F(x) - c*S*(x-1)]
        # For each x, the constraint is F(x) - c*S*(x-1) <= 1.0001
        # i.e., c*S*(x-1) >= F(x) - 1.0001

        # Let's compute the lower and upper bounds on c
        c_lower = 0.0
        c_upper = 10.0

        # For x > 1:
        # If S > 0: c >= (F(x) - 1.0001) / (S*(x-1))
        # If S < 0: c <= (F(x) - 1.0001) / (S*(x-1))  [note: S*(x-1) < 0, so inequality flips]

        # Let's compute F(x) for all x in [1, 10K]
        # This is expensive for large K, so let's be selective

        # Actually, let's just try a few c values for each K
        # c values to try: 0, 0.5, 1, 2, 5, 10 and the boundary value

        # First, let's find the boundary c where h(c) = 1.0001
        # This requires computing F(x) for all x, which is expensive

        # Let's use a binary search approach
        # But first, let's estimate the boundary

        # For large K, S = sum mu(k)/k converges to 0 (since sum mu(k)/k = 0 by PNT)
        # So for large K, the constraint is approximately max_x F(x) <= 1.0001
        # And max_x F(x) = max_x sum_{n<=x} g(n) where g(n) = sum_{k|n} mu(k)
        # For n <= K, g(n) = 1 if n = 1, 0 otherwise (since sum_{k|n} mu(k) = 0 for n > 1)
        # So F(x) = 1 for x >= 1, and max F(x) = 1
        # This means for large K, the constraint is 1 - c*S*(x-1) <= 1.0001
        # Which is always satisfied for c >= 0 if S >= 0, or for c <= 0 if S < 0

        # Wait, let me reconsider. For n > K, g(n) = sum_{k|n, k<=K} mu(k) which is not necessarily 0
        # So F(x) for x > K is not simply 1

        # Let's just try a few c values for each K
        c_values = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

        # Also try the boundary value
        # Estimate: if S is small, the boundary c is large
        # Let's just use the c_values

        for c in c_values:
            if time.time() > deadline - 60:
                break

            # Check feasibility: max_x [F(x) - c*S*(x-1)] <= 1.0001
            # We need to compute F(x) for all x in [1, 10K]
            # This is expensive, so let's only do it for promising c values

            # For now, let's just compute the objective and assume feasibility
            # We'll verify later
            a = c * obj_coeff

            if a > best_a:
                # Store this candidate for later verification
                results.append((K, c, a, max_F, S, H_K, obj_mu))

    # Sort results by a descending
    results.sort(key=lambda x: x[2], reverse=True)

    # Verify top candidates
    for K, c, a, max_F, S, H_K, obj_mu in results[:50]:
        if time.time() > deadline - 30:
            break

        max_x = 10 * K
        mu_K = mu_full[1:K + 1]

        # Compute F(x) for all x in [1, max_x]
        g = np.zeros(max_x + 1, dtype=np.float64)
        for k in range(1, K + 1):
            v = mu_K[k - 1]
            if v != 0:
                g[k::k] += v
        F = np.cumsum(g)

        # Compute M(x) = F(x) - c*S*(x-1)
        xs = np.arange(1, max_x + 1)
        M = F[1:max_x + 1] - c * S * (xs - 1)

        if np.max(M) > 1.0001:
            continue

        # Construct domain and values
        domain = np.arange(1, K + 1, dtype=np.int64)
        values = (c * mu_K).astype(np.float64)

        # Clip to [-10, 10]
        values = np.clip(values, -10.0, 10.0)

        # Compute actual objective
        # S_actual = sum values[k-1] / k
        S_actual = np.sum(values / domain.astype(np.float64))
        # f(1) adjustment: f(1) = values[0] - S_actual
        # g(k) = values[k-1] - S_actual for all k
        # a_actual = -sum g(k)*log(k)/k = -sum values[k-1]*log(k)/k + S_actual*H_K
        log_domain = np.log(domain.astype(np.float64))
        a_actual = -np.sum(values * log_domain / domain.astype(np.float64)) + S_actual * H_K

        if a_actual > best_a:
            best_a = a_actual
            best_domain = domain.copy()
            best_values = values.copy()

    # Fallback
    if best_domain is None:
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
    else:
        domain = best_domain
        values = best_values

    return domain, values


# EVOLVE-BLOCK-END
