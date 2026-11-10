# EVOLVE-BLOCK-START
import numpy as np
import time
import sys


def prime_number_theorem_partial_function():
    """
    Construct a partial function f from a subset of the positive integers to the reals
    that maximizes the objective a = -sum_k f(k)*log(k)/k subject to:
      * after subtracting S = sum f(k)/k from f(1), we have sum f(k)/k = 0
      * M(x) = sum_k f(k)*floor(x/k) <= 1 for all x in [1, 10K]
    where K = max(domain).
    
    Strategy: Use the Mobius function mu(k) which satisfies the identity
    sum_{d|n} mu(d) = 1, implying M(x) = sum_k mu(k)*floor(x/k) = 1 for all x >= 1.
    This gives the optimal constraint satisfaction. We include as many terms as possible
    within the time budget to maximize a = -sum mu(k)*log(k)/k -> 1 as K -> infinity.
    
    The key insight: mu(k) gives M(x)=1 exactly (not just <=1), so it's feasible.
    The objective approaches 1 from below as we include more terms.
    """
    start_time = time.time()
    deadline = 600.0  # seconds
    safety_margin = 30.0  # return 30s before the 630s kill
    
    def time_left():
        return deadline - safety_margin - (time.time() - start_time)
    
    # We'll build the Mobius function up to K and return (domain, values)
    # where values = mu(k) for k in domain.
    # 
    # The evaluator computes:
    #   1. Clip values to [-10, 10] (mu is in {-1, 0, 1}, so no clipping needed)
    #   2. S = sum_k f(k)/k; f(1) -= S; now sum f(k)/k = 0
    #   3. Check M(x) = sum f(k)*floor(x/k) <= 1.0001 for x in [1, 10K]
    #   4. a = -sum f(k)*log(k)/k
    #
    # For mu: sum_{k<=x} mu(k)*floor(x/k) = 1 for all x >= 1 (exact identity)
    # After the f(1) adjustment: f(1) becomes mu(1) - S = 1 - S
    # where S = sum_k mu(k)/k
    # New M(x) = (1-S)*floor(x/1) + sum_{k>=2} mu(k)*floor(x/k)
    #          = 1 - S*x + (1 - 1*x) ... wait let me reconsider.
    #
    # Actually: M(x) = sum_k f(k)*floor(x/k)
    # Original: M_orig(x) = sum_k mu(k)*floor(x/k) = 1
    # After adjustment: f(1) = 1 - S, others unchanged
    # M_new(x) = (1-S)*floor(x/1) + sum_{k>=2} mu(k)*floor(x/k)
    #          = (1-S)*x + (1 - 1*x)  [since sum_{k>=1} mu(k)*floor(x/k) = 1]
    #          = x - S*x + 1 - x = 1 - S*x
    #
    # So M_new(x) = 1 - S*x where S = sum_{k=1}^{K} mu(k)/k
    # For large K, S -> 0 (since sum mu(k)/k converges to 0 by PNT)
    # But for finite K, S is small but nonzero.
    # M_new(x) = 1 - S*x for x in [1, 10K]
    # If S > 0: max M_new is at x=1: M_new(1) = 1 - S < 1. OK.
    #           min M_new is at x=10K: M_new(10K) = 1 - 10K*S
    #           We need M_new(x) <= 1.0001, which is always true if S >= 0.
    # If S < 0: max M_new is at x=10K: M_new(10K) = 1 - 10K*S = 1 + 10K*|S|
    #           We need 1 + 10K*|S| <= 1.0001, i.e., |S| <= 0.0001/(10K)
    #
    # For K=200000: |S| <= 0.0001/2000000 = 5e-11. That's very tight!
    #
    # Actually sum_{k=1}^K mu(k)/k is approximately 1/log(K) * (something) ...
    # No, sum mu(k)/k converges to 0. The error is O(1/log K) or similar.
    # For K=200000, log(K) ≈ 12.2, so S might be ~ 0.08 or so? That's way too big.
    #
    # Let me reconsider. The sum mu(k)/k for k=1..K:
    # mu(1)/1 + mu(2)/2 + mu(3)/3 + ... 
    # = 1 - 1/2 - 1/3 - 1/5 + 1/6 - 1/7 - 1/11 + 1/10 + 1/13 - ...
    # This is related to 1/zeta(s) at s=1, which diverges.
    # Actually sum mu(k)/k = 0 (in the sense of the limit), but partial sums oscillate.
    #
    # The partial sum S_K = sum_{k=1}^K mu(k)/k.
    # By PNT, S_K = O(1/log K) roughly? Let's estimate for K=200000.
    # Actually it's known that sum_{k<=x} mu(k)/k = O(1/log x) is not quite right.
    # The sum mu(k)/k = 1/zeta(1) which diverges... no.
    # 
    # Wait: sum_{k=1}^inf mu(k)/k = 0 (this is 1/zeta(1) in a regularized sense,
    # but actually it converges conditionally to 0).
    # 
    # The partial sums S_K oscillate and decay. For K=200000, S_K is probably
    # on the order of 0.01-0.1.
    #
    # If S_K = 0.05 and K = 200000: M_new(10K) = 1 - 10*200000*0.05 = 1 - 100000 < 0.
    # That's fine for the <= 1.0001 constraint! The constraint is M(x) <= 1.0001,
    # not M(x) >= something. So negative values are fine.
    #
    # If S_K = -0.05: M_new(10K) = 1 + 10*200000*0.05 = 1 + 100000 > 1.0001. INFEASIBLE!
    #
    # So we need S_K >= 0, or |S_K| very small.
    #
    # Alternative: Don't use pure mu. Use mu but adjust to make S = 0 exactly
    # by choosing the domain carefully, or use a modified function.
    #
    # Better approach: Use f(k) = mu(k) for k in some set, and choose the set
    # such that S = sum f(k)/k is close to 0 or positive.
    #
    # Even better: The constraint is M(x) <= 1.0001. With mu, M_new(x) = 1 - S*x.
    # If S > 0: M_new(x) <= 1 for all x. Feasible!
    # If S < 0: M_new(x) = 1 + |S|*x, which exceeds 1.0001 when x > 0.0001/|S|.
    #
    # So we need S >= 0. Let's compute S for various K and pick K where S >= 0.
    #
    # Actually, let me reconsider the problem. We can choose ANY subset of positive
    # integers as the domain. We don't have to use all of 1..K.
    #
    # The optimal solution in the limit is f(k) = mu(k) for all k, giving a -> 1.
    # But the S adjustment creates a problem.
    #
    # Key insight: If we include k=1 in the domain, f(1) gets adjusted.
    # If we DON'T include k=1, then f(1) = 0 (by default), and S = sum f(k)/k.
    # Then f(1) = 0 - S = -S. But 1 is not in the domain, so this doesn't apply!
    #
    # Re-reading the task: "f(1) = 0 before this if 1 is not in the domain"
    # So if 1 is not in domain: f(1) starts as 0, then f(1) = 0 - S = -S.
    # But 1 is still not in the domain! So f(1) is never actually used in M(x)?
    #
    # Wait, I think the adjustment is: conceptually extend f to include 1 if needed,
    # adjust f(1), then compute M(x) with the adjusted values.
    #
    # Let me re-read: "f(1) is replaced by f(1) - S with S = sum_k f(k)/k over the
    # clipped values (f(1) = 0 before this if 1 is not in the domain)"
    #
    # So the procedure is:
    # 1. Start with our f defined on domain D.
    # 2. If 1 not in D, set f(1) = 0 conceptually.
    # 3. Compute S = sum over all k in (D union {1}) of f(k)/k.
    #    Wait no: S = sum_k f(k)/k over the clipped values. The sum is over the domain.
    #    But f(1) might be 0 if 1 not in domain.
    #    
    #    I think: S = sum_{k in D} f(k)/k. If 1 not in D, f(1) is treated as 0
    #    for the purpose of the adjustment, but 1 is not added to the domain.
    #    
    #    Actually: "f(1) is replaced by f(1) - S" - this implies 1 IS in the domain
    #    after the adjustment. So the evaluator adds 1 to the domain if needed.
    #
    # Let me assume: the effective domain is D' = D union {1}, with f(1) = original
    # f(1) if 1 in D, else 0. Then f(1) = f(1) - S where S = sum_{k in D'} f(k)/k
    # computed BEFORE the adjustment. Then M(x) uses D' with the adjusted f(1).
    #
    # With mu: D = {1, 2, ..., K}, f(k) = mu(k).
    # S = sum_{k=1}^K mu(k)/k.
    # f(1) adjusted to 1 - S.
    # M(x) = (1-S)*x + sum_{k=2}^K mu(k)*floor(x/k) = 1 - S*x (as computed above).
    #
    # For feasibility: need M(x) <= 1.0001 for all x in [1, 10K].
    # If S >= 0: M(x) = 1 - S*x <= 1. Always feasible.
    # If S < 0: M(x) = 1 + |S|*x. Max at x=10K: 1 + 10K*|S| <= 1.0001
    #           => |S| <= 0.0001/(10K) = 1e-8/K.
    #
    # So we need S >= 0 or |S| extremely small.
    #
    # Strategy: Compute mu(k) for k=1..K, compute S = sum mu(k)/k.
    # If S >= 0: use all K terms. Objective a = -sum mu(k)*log(k)/k.
    # If S < 0: we can try removing some terms to make S >= 0, or use a smaller K.
    #
    # Actually, a simpler approach: just try increasing K and check if S >= 0.
    # The sum mu(k)/k oscillates, so for some K it will be positive.
    #
    # Let's compute mu and S for various K and find the best feasible one.
    
    def compute_mobius(K):
        """Compute Mobius function mu(1)..mu(K) using sieve."""
        mu = np.ones(K + 1, dtype=np.int8)
        mu[0] = 0
        is_prime = np.ones(K + 1, dtype=bool)
        primes = []
        for i in range(2, K + 1):
            if is_prime[i]:
                primes.append(i)
                mu[i] = -1
                for j in primes:
                    if i * j > K:
                        break
                    is_prime[i * j] = False
                    if i % j == 0:
                        mu[i * j] = 0
                        break
                    else:
                        mu[i * j] = -mu[i]
        return mu
    
    # Compute Mobius up to a reasonable K
    # Start with K=200000 as the maximum allowed
    K_max = 200000
    
    if time_left() < 10:
        # Not enough time, return a small feasible solution
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    mu = compute_mobius(K_max)
    
    # Compute S(K) = sum_{k=1}^K mu(k)/k for all K
    # S(K) = S(K-1) + mu(K)/K
    inv_k = np.zeros(K_max + 1, dtype=np.float64)
    for k in range(1, K_max + 1):
        inv_k[k] = 1.0 / k
    
    mu_float = mu.astype(np.float64)
    terms = mu_float * inv_k  # mu(k)/k
    S_cumsum = np.cumsum(terms)  # S_cumsum[K] = sum_{k=1}^K mu(k)/k
    
    # Find the largest K where S(K) >= 0 (feasible)
    # S_cumsum[0] is undefined, S_cumsum[k] = S(k) for k >= 1
    feasible_Ks = np.where(S_cumsum >= 0)[0]
    feasible_Ks = feasible_Ks[feasible_Ks >= 1]
    
    if len(feasible_Ks) == 0:
        # No feasible K found with pure mu. Use the baseline.
        domain = np.array([1, 2, 3, 5, 30], dtype=np.int64)
        values = np.array([3.47155351603266, -1.0, -1.0, -1.0, 1.0], dtype=np.float64)
        return domain, values
    
    # We want to maximize a = -sum_{k=1}^K mu(k)*log(k)/k
    # This is -sum mu(k)*log(k)/k = sum (-mu(k))*log(k)/k
    # For mu: a(K) = -sum_{k=1}^K mu(k)*log(k)/k
    # As K increases, a(K) -> 1 (the optimal value).
    # So we want the largest feasible K.
    
    best_K = feasible_Ks[-1]  # largest K with S(K) >= 0
    
    # But we should also check: does a larger K give better a even if slightly infeasible?
    # No, infeasible scores 0. So we need the largest FEASIBLE K.
    #
    # However, we might be able to do better by not using ALL terms up to K.
    # E.g., use mu(k) for k in some subset that makes S >= 0 while including
    # many large-k terms (which contribute more to the objective).
    #
    # For now, let's use the full range 1..best_K.
    
    # Compute the objective for this K
    log_k = np.zeros(K_max + 1, dtype=np.float64)
    for k in range(2, K_max + 1):
        log_k[k] = np.log(k)
    
    obj_terms = mu_float * log_k * inv_k  # mu(k)*log(k)/k
    a_cumsum = np.cumsum(obj_terms)  # sum mu(k)*log(k)/k
    a_val = -a_cumsum[best_K - 1]  # a = -sum mu(k)*log(k)/k
    
    # Build domain and values
    # Domain: all k from 1 to best_K where mu(k) != 0 (to keep m small)
    # Actually, we can include all k; mu(k)=0 terms don't affect anything.
    # But to reduce m, only include k where mu(k) != 0.
    
    nonzero_mask = mu[1:best_K + 1] != 0
    domain = np.where(nonzero_mask)[0] + 1
    values = mu[1:best_K + 1][nonzero_mask].astype(np.float64)
    
    # Check time
    if time_left() < 5:
        # Return what we have
        return domain, values
    
    # Verify feasibility (spot check)
    # M(x) = 1 - S*x where S = S_cumsum[best_K-1]
    S_val = S_cumsum[best_K - 1]
    # Max M(x) for x in [1, 10*best_K]:
    if S_val >= 0:
        max_M = 1.0  # at x=1
    else:
        max_M = 1.0 + 10.0 * best_K * abs(S_val)
    
    if max_M > 1.0001:
        # Shouldn't happen since we checked S >= 0, but just in case
        # Fall back to smaller K
        for K_try in feasible_Ks[::-1]:
            S_try = S_cumsum[K_try - 1]
            if S_try >= 0:
                best_K = K_try
                break
            max_M_try = 1.0 + 10.0 * K_try * abs(S_try)
            if max_M_try <= 1.0001:
                best_K = K_try
                break
    
    # Rebuild with final best_K
    nonzero_mask = mu[1:best_K + 1] != 0
    domain = np.where(nonzero_mask)[0] + 1
    values = mu[1:best_K + 1][nonzero_mask].astype(np.float64)
    
    return domain, values


# EVOLVE-BLOCK-END
