# EVOLVE-BLOCK-START
import time
import numpy as np


def _mobius_up_to(n):
    """Compute the Mobius function mu(k) for k = 1..n using a sieve."""
    mu = np.ones(n + 1, dtype=np.float64)
    is_prime = np.ones(n + 1, dtype=bool)
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1.0
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                mu[i * p] = 0.0
                break
            else:
                mu[i * p] = -mu[i]
    return mu


def _compute_M_max_fast(domain, values, K):
    """
    Compute max of M(x) = sum_k f(k)*floor(x/k) for x in [1, 10K].
    Uses the fact that M(x) is piecewise linear with slope changes at multiples.
    We track the running sum and only need to check at critical points.
    
    Key insight: M(x+1) - M(x) = sum_k f(k) * (floor((x+1)/k) - floor(x/k))
    The difference is nonzero only when k divides x+1.
    So M(x) is a step function that changes only at x values that are multiples of some k in domain.
    
    More efficiently: M(x) = sum_k f(k) * floor(x/k).
    For x in [1, 10K], we can compute this by noting that for each k,
    floor(x/k) increases by 1 at each multiple of k.
    
    We'll use an event-based approach: create an array of deltas.
    """
    m = len(domain)
    if m == 0:
        return 0.0
    x_max = 10 * K
    
    # M(x) = sum_k f(k) * floor(x/k)
    # floor(x/k) = number of multiples of k in [1, x]
    # So M(x) = sum_k f(k) * (number of multiples of k in [1, x])
    # = sum_{j: j*k <= x} f(k) for each k
    # = sum_{n <= x} (sum_{k|n} f(k))
    # Let g(n) = sum_{k|n} f(k), then M(x) = sum_{n=1}^{x} g(n)
    
    # Compute g(n) for n up to x_max using a sieve-like approach
    # g(n) = sum_{k|n, k in domain} f(k)
    # We can do: for each k in domain, add f(k) to g(k), g(2k), g(3k), ...
    
    g = np.zeros(x_max + 1, dtype=np.float64)
    for k, f in zip(domain, values):
        if f == 0.0:
            continue
        k = int(k)
        # Add f to all multiples of k up to x_max
        g[k::k] += f
    
    # M(x) = cumulative sum of g
    M = np.cumsum(g)
    # M[0] = 0, M[1] = g(1), M[x] = sum_{n=1}^{x} g(n)
    # We want max of M(x) for x in [1, 10K]
    if x_max < 1:
        return 0.0
    return float(np.max(M[1:x_max+1]))


def _compute_a(domain, values):
    """Compute a = -sum_k f(k)*log(k)/k."""
    if len(domain) == 0:
        return 0.0
    k = domain.astype(np.float64)
    f = values
    return float(-np.sum(f * np.log(k) / k))


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes a hidden number-theoretic objective related to the Prime Number Theorem.

    Returns:
        (domain, values): a tuple of two numpy arrays of equal length describing f.
    """
    t_start = time.time()
    DEADLINE = 600.0  # evaluator kills at 630s; keep safety margin

    def time_left():
        return DEADLINE - (time.time() - t_start)

    # Fallback: the seed's known-feasible construction.
    fallback_domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
    fallback_values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

    best_domain = fallback_domain.copy()
    best_values = fallback_values.copy()
    best_a = _compute_a(best_domain, best_values)

    # Strategy: use the Mobius function mu(k) for k = 1..N. 
    # The full Mobius function satisfies sum_{k|x} mu(k) = 0 for x>1, and
    #   sum_{k<=x} mu(k) floor(x/k) = 1  for all x >= 1.
    # For a partial function using mu(k) for k <= N, M(x) = 1 for x <= N.
    # For x > N, M(x) can deviate. We need M(x) <= 1.0001 for x in [1, 10N].
    
    # Try increasing N, checking feasibility with the fast method.
    candidate_Ns = [200, 500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000, 50000, 75000, 100000]
    
    for N in candidate_Ns:
        if time_left() < 30.0:
            break
        
        mu = _mobius_up_to(N)
        # domain = all k in 1..N with mu(k) != 0
        mask = mu[1:] != 0.0
        domain = np.nonzero(mask)[0] + 1
        values = mu[1:][mask]
        K = int(domain.max()) if len(domain) > 0 else 1
        
        if K > 200000:
            continue
            
        # Check feasibility: max M(x) over x in [1, 10K]
        Mmax = _compute_M_max_fast(domain, values, K)
        
        if Mmax <= 1.0001:
            a = _compute_a(domain, values)
            if a > best_a:
                best_a = a
                best_domain = domain.copy()
                best_values = values.copy()
        else:
            # Try damped version: multiply values by a factor < 1 to reduce M(x)
            # M(x) scales linearly with values, so if Mmax > 1, scale by 1.0001/Mmax
            # But this also reduces a. Only useful if the unscaled is infeasible.
            # Actually, we want to maximize a = -sum f(k) log(k)/k, so scaling down reduces |a|.
            # Better to just skip infeasible N and try smaller ones.
            pass

    # Also try: use mu(k) for k in a sparse set (e.g., only squarefree numbers up to N)
    # This is already what we do since mu(k)=0 for non-squarefree.
    
    # Try another approach: use f(k) = -mu(k) for k up to N (flipped sign)
    # This might give different feasibility characteristics.
    # Actually, let's try: f(k) = mu(k) for k up to N, but also add a correction at k=1.
    # The normalization step adjusts f(1) so that sum f(k)/k = 0.
    # Let's see what the normalized version looks like.
    
    # For the Mobius construction, sum_{k=1}^{N} mu(k)/k ≈ 0 (it oscillates).
    # So the normalization doesn't change much.
    
    # Let's also try: f(k) = mu(k) * log(k) / k^2 or similar weightings to optimize a.
    # a = -sum f(k) log(k)/k. If f(k) = mu(k), then a = -sum mu(k) log(k)/k.
    # By PNT-related results, sum_{k<=N} mu(k) log(k)/k ≈ -1 (related to log zeta).
    # So a ≈ 1 for large N. That's not great.
    
    # Alternative: f(k) = -mu(k) for k in some set. Then a = sum mu(k) log(k)/k ≈ -1.
    # Still not great.
    
    # The key insight: we want to maximize a = -sum f(k) log(k)/k subject to M(x) <= 1.
    # M(x) = sum f(k) floor(x/k) <= 1 for all x in [1, 10K].
    # At x = K: M(K) = sum_{k<=K} f(k) floor(K/k) <= 1.
    # If f(k) = mu(k), M(K) = 1 exactly.
    # The constraint is tightest at certain x values.
    
    # Let's try: f(k) = c * mu(k) for k up to N, where c is chosen so that max M(x) = 1.
    # Then a = -c * sum mu(k) log(k)/k.
    # If c < 1, a is smaller. If c > 1, M(x) > 1 somewhere.
    # So the best we can do with pure mu is c = 1, giving a = -sum mu(k) log(k)/k.
    
    # To do better, we need a different f. Consider f(k) = mu(k) * h(k) where h(k) is a weight.
    # Then M(x) = sum mu(k) h(k) floor(x/k) and a = -sum mu(k) h(k) log(k)/k.
    # We want to maximize a subject to M(x) <= 1.
    
    # This is a linear programming problem! Let's try a simple heuristic:
    # Start with f(k) = mu(k), then greedily increase f(k) for k with large log(k)/k
    # while maintaining M(x) <= 1.
    
    # Greedy approach:
    # 1. Start with f = 0.
    # 2. For k from 1 to N, try to set f(k) as large as possible (positive or negative)
    #    while maintaining feasibility.
    
    # Actually, let's try: f(k) = mu(k) for k up to N, but only for k in a carefully chosen subset.
    # Or: f(k) = mu(k) * w(k) where w(k) is optimized.
    
    # Given time constraints, let's try a few more N values with the fast method.
    
    # Try larger N values if time permits
    large_Ns = [150000, 200000]
    for N in large_Ns:
        if time_left() < 60.0:
            break
        
        mu = _mobius_up_to(N)
        mask = mu[1:] != 0.0
        domain = np.nonzero(mask)[0] + 1
        values = mu[1:][mask]
        K = int(domain.max()) if len(domain) > 0 else 1
        
        if K > 200000:
            continue
            
        Mmax = _compute_M_max_fast(domain, values, K)
        
        if Mmax <= 1.0001:
            a = _compute_a(domain, values)
            if a > best_a:
                best_a = a
                best_domain = domain.copy()
                best_values = values.copy()

    # Safety: ensure the returned result is feasible. Re-verify the best candidate.
    if len(best_domain) > 0:
        K = int(best_domain.max())
        Mmax = _compute_M_max_fast(best_domain, best_values, K)
        if Mmax > 1.0001:
            # Fallback to the known-feasible seed.
            best_domain = fallback_domain.copy()
            best_values = fallback_values.copy()

    # Final safety: ensure domain/values are valid (distinct positive ints, K <= 200000).
    if len(best_domain) == 0:
        best_domain = fallback_domain.copy()
        best_values = fallback_values.copy()
    if int(best_domain.max()) > 200000:
        best_domain = fallback_domain.copy()
        best_values = fallback_values.copy()

    return best_domain, best_values


# EVOLVE-BLOCK-END
