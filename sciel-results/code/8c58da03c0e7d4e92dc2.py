"""Incentive-compatible finite auction design via Myerson optimal auction."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue
    using the Myerson optimal auction approach.
    
    Args:
        distribution_spec: Dict with 'probs' (list of 16 floats) or similar structure.
        support: List of (v1, v2) tuples, likely [(0,0), ..., (3,3)].
        constraints: Dict with constraints (may be empty).
    
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        probs = np.array(distribution_spec.get('probs', [1/16]*16), dtype=float)
    else:
        probs = np.array(distribution_spec, dtype=float)
    
    probs = probs / probs.sum()
    
    # Support: all (v1, v2) for v1, v2 in {0,1,2,3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = 16
    
    # Map type to index
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # For Myerson optimal auction with a single bidder and multiple goods:
    # We need to find the allocation rule that maximizes expected virtual surplus.
    # 
    # Virtual value for type v = (v1, v2):
    # phi(v) = v1 + (1 - F1(v1)) / f1(v1) + v2 + (1 - F2(v2)) / f2(v2)
    # where F1, f1 are marginal CDF/PDF for good 1, and F2, f2 for good 2.
    #
    # However, since values are discrete and the distribution may not be independent,
    # we need to be more careful. Let's use the general approach:
    #
    # For a single bidder, the optimal mechanism assigns good i to the bidder
    # with probability x_i(v) such that the expected virtual surplus is maximized.
    #
    # For discrete distributions, the virtual value for type v with respect to good i is:
    # phi_i(v) = v_i + sum_{t > v_i} P(V_i = t | V_i >= v_i) / P(V_i = v_i) * (P(V_i >= t) - P(V_i >= v_i))
    # 
    # Actually, let's use a simpler approach: enumerate all possible allocation rules
    # that are monotone and incentive-compatible, and pick the one with highest expected revenue.
    #
    # For a single additive bidder, an allocation rule is IC iff it is monotone:
    # if v >= v' component-wise, then x(v) >= x'(v) component-wise.
    #
    # The payment for type v is:
    # p(v) = v·x(v) - sum_{t < v} x(t) (where t < v means component-wise)
    #
    # Let's enumerate monotone allocation rules. For each good, x_i(v) must be monotone
    # in both v1 and v2.
    
    # First, compute marginal distributions
    # P(V1 = v1) for v1 in {0,1,2,3}
    p_v1 = np.zeros(4)
    p_v2 = np.zeros(4)
    for v_idx, (v1, v2) in enumerate(types):
        p_v1[v1] += probs[v_idx]
        p_v2[v2] += probs[v_idx]
    
    # CDFs
    F1 = np.cumsum(p_v1)
    F2 = np.cumsum(p_v2)
    
    # Virtual values for each good
    # For discrete case: phi_i(v_i) = v_i + (1 - F_i(v_i)) / f_i(v_i) if f_i(v_i) > 0, else -inf
    phi1 = np.zeros(4)
    phi2 = np.zeros(4)
    for i in range(4):
        if p_v1[i] > 1e-15:
            phi1[i] = i + (1 - F1[i]) / p_v1[i]
        else:
            phi1[i] = -np.inf
        if p_v2[i] > 1e-15:
            phi2[i] = i + (1 - F2[i]) / p_v2[i]
        else:
            phi2[i] = -np.inf
    
    # For the optimal mechanism, we allocate good 1 with probability 1 if phi1(v1) > 0, else 0
    # Similarly for good 2. But this is for independent distributions.
    #
    # For the general case, we need to consider joint types. Let's use the fact that
    # for a single bidder, the optimal allocation is:
    # x_i(v) = 1 if the virtual surplus of allocating good i is positive.
    #
    # Virtual surplus of allocating good 1 to type v:
    # E[phi1(V1) | V2 = v2, V1 >= some threshold]
    #
    # Actually, let's use a direct approach: for each possible monotone allocation,
    # compute the expected revenue and pick the best.
    #
    # Since values are in {0,1,2,3}, we can enumerate all monotone allocation rules.
    # A monotone allocation for good 1: x1(v1, v2) must be non-decreasing in v1 and v2.
    # So x1(0,0) <= x1(1,0) <= x1(2,0) <= x1(3,0)
    # and x1(0,0) <= x1(0,1) <= x1(0,2) <= x1(0,3)
    # etc.
    #
    # This is a large space, but we can use the fact that for the optimal mechanism,
    # the allocation is a "threshold" rule: allocate good i if v_i >= threshold_i(v_{-i}).
    
    # Let's try a simpler approach: assume the allocation depends only on the respective value.
    # x1(v1, v2) = a(v1) and x2(v1, v2) = b(v2)
    # where a and b are monotone non-decreasing functions from {0,1,2,3} to [0,1].
    
    # Enumerate all monotone a: a(0) <= a(1) <= a(2) <= a(3)
    # Each a(i) can be in [0,1], but for the optimal mechanism, it's often 0 or 1.
    # Let's first try binary allocations: a(i) in {0,1} with a(0) <= a(1) <= a(2) <= a(3).
    
    def enumerate_monotone_binary():
        """Enumerate all monotone binary functions from {0,1,2,3} to {0,1}."""
        results = []
        for a0 in [0, 1]:
            for a1 in range(a0, 2):
                for a2 in range(a1, 2):
                    for a3 in range(a2, 2):
                        results.append([a0, a1, a2, a3])
        return results
    
    monotone_a = enumerate_monotone_binary()
    monotone_b = enumerate_monotone_binary()
    
    best_revenue = -np.inf
    best_a = None
    best_b = None
    
    for a in monotone_a:
        for b in monotone_b:
            # Compute expected revenue
            revenue = 0.0
            for v_idx, (v1, v2) in enumerate(types):
                pi = probs[v_idx]
                x1_v = a[v1]
                x2_v = b[v2]
                # Payment: p(v) = v1*x1(v) + v2*x2(v) - sum_{t<v1} x1(t,v2) - sum_{t<v2} x2(v1,t)
                # For our simplified model: sum_{t<v1} x1(t,v2) = sum_{t=0}^{v1-1} a(t)
                # sum_{t<v2} x2(v1,t) = sum_{t=0}^{v2-1} b(t)
                p_v = v1 * x1_v + v2 * x2_v - np.sum(a[:v1]) - np.sum(b[:v2])
                if p_v >= -1e-10:  # IR check
                    revenue += pi * max(0, p_v)
                else:
                    # IR violated, this allocation is not feasible
                    break
            else:
                if revenue > best_revenue:
                    best_revenue = revenue
                    best_a = a
                    best_b = b
    
    # If no feasible binary allocation found, use a = b = [0,0,0,0] (no allocation)
    if best_a is None:
        best_a = [0, 0, 0, 0]
        best_b = [0, 0, 0, 0]
    
    # Now, let's also try the Myerson threshold approach more carefully
    # For each good, find the optimal threshold
    
    # Compute virtual values properly for the joint distribution
    # For good 1: phi1(v1, v2) = E[V1 | V1 >= v1, V2 = v2] + (P(V1 >= v1+1, V2 = v2) - P(V1 >= v1, V2 = v2)) / P(V1 = v1, V2 = v2) * something
    #
    # Actually, let's use the correct formula for discrete virtual values:
    # phi(v) = v - (1 - F(v)) / f(v) for continuous, but for discrete:
    # phi(v_i) = v_i + sum_{t > v_i} (F(t-1) - F(v_i)) / f(v_i) * (t - v_i)
    #
    # Let's just use the LP approach as a fallback if the analytical approach doesn't work well.
    
    # Build the allocation and payment vectors
    allocations = np.zeros((n_types, 2))
    for v_idx, (v1, v2) in enumerate(types):
        allocations[v_idx, 0] = best_a[v1]
        allocations[v_idx, 1] = best_b[v2]
    
    # Compute payments
    payments = np.zeros(n_types)
    for v_idx, (v1, v2) in enumerate(types):
        x1_v = best_a[v1]
        x2_v = best_b[v2]
        payments[v_idx] = v1 * x1_v + v2 * x2_v - np.sum(best_a[:v1]) - np.sum(best_b[:v2])
    
    # Ensure nonnegative payments
    payments = np.maximum(payments, 0)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
