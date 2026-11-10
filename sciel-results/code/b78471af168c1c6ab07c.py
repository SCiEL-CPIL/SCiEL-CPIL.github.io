# EVOLVE-BLOCK-START
import numpy as np
import time


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k)*log(k)/k subject to:
      - sum_k f(k)/k == 0 (enforced by the evaluator via f(1) adjustment)
      - M(x) = sum_k f(k) * floor(x/k) <= 1.0001 for all x in [1, 10*K]
    where K is the maximum key in the domain.

    The optimal infinite solution is the Mobius function mu(k). We compute mu(k) for
    k up to K, then solve a constrained optimization to maximize a subject to the
    linear constraints M(x) <= 1.0001 and the sum constraint (handled by the evaluator).

    Strategy:
      1. Compute mu(k) for k = 1..K using a sieve.
      2. The raw mu gives sum mu(k) floor(x/k) = 1 for all x, so M(x) = 1 (feasible).
      3. To improve the objective, we can try to adjust values while keeping feasibility.
      4. Use a simple iterative refinement: start with mu, then try small perturbations
         that increase a while maintaining M(x) <= 1.0001.

    However, a cleaner approach: the objective is linear in f, and the constraints are
    linear. This is a linear program. But with K up to 200000 and 10K constraints,
    it's too large for standard LP solvers in the time budget.

    Better approach: exploit the structure. The constraint M(x) <= 1 for all x in [1,10K]
    is equivalent to: for each x, sum_{k|x, k<=K} f(k) * (x//k) ... no, it's
    M(x) = sum_k f(k) * floor(x/k).

    Key insight: If we take f(k) = mu(k) for all k <= K, then M(x) = 1 for all x <= K,
    and for x > K, M(x) = sum_{k<=K} mu(k) floor(x/k) which is approximately 1 (the
    Mertens function M(x) is small). In fact, sum_{k<=x} mu(k) floor(x/k) = 1 exactly
    for all x, so truncating at K gives M(x) = 1 - sum_{k>K, k|x... } which is close to 1.

    The objective with mu: a = -sum_{k=1}^K mu(k) log(k)/k. This converges to 1 as K->inf.
    For finite K, it's less than 1. To get above 1, we need to go beyond just mu.

    Actually, the score is value/seed where seed=1.0. The current program scores 1.0.
    To beat 1.0, we need a > 1.0.

    With mu(k) for k=1..K: a = -sum mu(k) log(k)/k. For K=200000, this is very close to 1
    but slightly less. We need to find a modification that pushes it above 1.

    One idea: the evaluator clips values to [-10,10], and mu is in {-1,0,1}, so no clipping.
    The f(1) adjustment: S = sum f(k)/k, then f(1) -= S. This changes the objective.

    Let's compute: with mu, sum mu(k)/k for k=1..K is small (converges to 0). So S is small,
    and f(1) becomes mu(1) - S = 1 - S. The new objective is
    a = -(1-S)log(1)/1 - sum_{k>=2} mu(k) log(k)/k = -sum_{k>=2} mu(k) log(k)/k
    since log(1)=0. So the f(1) adjustment doesn't change the objective when log(1)=0!

    So a = -sum_{k=2}^K mu(k) log(k)/k. This is the partial sum of the series for 1.
    The full series sum_{k=1}^inf mu(k) log(k)/k = -1 (related to zeta(1) divergence... actually
    sum mu(k) log(k)/k = -1 from the identity -zeta'(1)/zeta(1) = 1, but it's conditionally convergent).

    For finite K, -sum_{k=2}^K mu(k) log(k)/k < 1. We need to add terms or adjust to exceed 1.

    Approach: Use mu(k) for k <= K, and add a small positive value to some f(k) to boost the
    objective while keeping M(x) <= 1.0001. The slack is 0.0001 in M(x).

    Let's implement: compute mu, check feasibility, then do a simple greedy improvement.
    """
    t0 = time.time()
    deadline = t0 + 600  # 600s deadline

    # Step 1: Compute mu(k) for k = 1..K
    K = 200000
    mu = np.ones(K + 1, dtype=np.int8)
    mu[0] = 0
    is_prime = np.ones(K + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, K + 1):
        if is_prime[i]:
            primes.append(i)
            mu[i] = -1
            for j in range(i * i, K + 1, i):
                is_prime[j] = False
        for p in primes:
            if p * i > K:
                break
            if i % p == 0:
                mu[i * p] = 0
                break
            else:
                mu[i * p] = -mu[i]

    # Build domain and values from mu (only nonzero mu)
    mask = mu[1:] != 0
    domain = np.where(mask)[0] + 1
    values = mu[1:][mask].astype(np.float64)

    # Step 2: Check feasibility and compute objective
    def check_and_score(domain, values, t0, deadline):
        """Check M(x) <= 1.0001 for x in [1, 10*K] and compute objective."""
        if len(domain) == 0:
            return -np.inf, False
        K_local = int(domain.max())
        max_x = 10 * K_local
        # Compute M(x) = sum_k f(k) * floor(x/k) for x in [1, max_x]
        # Efficient: for each k, f(k) * floor(x/k) contributes to M(x) for x in [1, max_x]
        # Use the fact that floor(x/k) is piecewise constant
        M = np.zeros(max_x + 1, dtype=np.float64)
        for k, f in zip(domain, values):
            k = int(k)
            f = float(f)
            if f == 0:
                continue
            # floor(x/k) for x in [1, max_x]
            # For x in [j*k, (j+1)*k - 1], floor(x/k) = j
            # j ranges from 0 to max_x // k
            # M[x] += f * j for x in [j*k, min((j+1)*k-1, max_x)]
            for j in range(1, max_x // k + 1):
                start = j * k
                end = min((j + 1) * k - 1, max_x)
                M[start:end + 1] += f * j
        # Check feasibility
        if M[1:].max() > 1.0001:
            return -np.inf, False
        # Compute S = sum f(k)/k
        S = np.sum(values / domain.astype(np.float64))
        # Adjust f(1)
        if 1 in domain:
            idx1 = np.where(domain == 1)[0][0]
            values_adj = values.copy()
            values_adj[idx1] -= S
        else:
            values_adj = values.copy()
            # f(1) = 0 - S = -S, but 1 not in domain, so we need to add it? 
            # The evaluator says "f(1) = 0 before this if 1 is not in the domain"
            # So we don't need to add 1 to domain; the evaluator handles it.
            pass
        # Compute objective a = -sum f(k) log(k) / k
        a = -np.sum(values_adj * np.log(domain.astype(np.float64)) / domain.astype(np.float64))
        return a, True

    # Quick check
    a, feasible = check_and_score(domain, values, t0, deadline)
    if not feasible:
        # Fallback to the seed
        return np.array([1, 2, 3, 5, 30], dtype=np.int64), \
               np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)

    # Step 3: Try to improve by adding small perturbations
    # The slack in M(x) is 1.0001 - M(x). We can add delta to some f(k) if it doesn't
    # violate any M(x) <= 1.0001.
    # Objective change: da = -delta * log(k) / k for adding delta to f(k).
    # To increase a, we want delta * log(k) / k < 0, i.e., delta < 0 if log(k)/k > 0 (k>1).
    # So we should DECREASE f(k) for k > 1 to increase a.
    # But decreasing f(k) by delta > 0 changes M(x) by -delta * floor(x/k) for each x.
    # This makes M(x) smaller, so it's always feasible!
    # Wait: if we decrease f(k), M(x) decreases, so feasibility is preserved.
    # And a increases by delta * log(k)/k > 0 for k > 1.
    # So we can decrease f(k) arbitrarily? No, values are clipped to [-10, 10].
    # So we can decrease f(k) down to -10.

    # For k with mu(k) = 1, we can decrease from 1 to -10, delta = 11.
    # For k with mu(k) = -1, we can decrease from -1 to -10, delta = 9.
    # But wait, the f(1) adjustment: S changes, which affects f(1).
    # This is getting complex. Let's just set f(k) = -10 for all k > 1 where mu(k) != 0,
    # and see what happens.

    # Actually, let's be more careful. Set values to -10 for all k in domain, k > 1.
    values_new = values.copy()
    for i, k in enumerate(domain):
        if k > 1:
            values_new[i] = -10.0
    a_new, feasible_new = check_and_score(domain, values_new, t0, deadline)
    if feasible_new and a_new > a:
        values = values_new
        a = a_new

    # Even better: set f(k) = -10 for ALL k in domain (including k=1).
    values_all_neg10 = np.full(len(domain), -10.0, dtype=np.float64)
    a_all, feasible_all = check_and_score(domain, values_all_neg10, t0, deadline)
    if feasible_all and a_all > a:
        values = values_all_neg10
        a = a_all

    # Try: f(k) = -10 for k > 1, f(1) = 10
    values_mixed = values.copy()
    for i, k in enumerate(domain):
        if k == 1:
            values_mixed[i] = 10.0
        else:
            values_mixed[i] = -10.0
    a_mixed, feasible_mixed = check_and_score(domain, values_mixed, t0, deadline)
    if feasible_mixed and a_mixed > a:
        values = values_mixed
        a = a_mixed

    # Return the best found
    return domain, values


# EVOLVE-BLOCK-END
