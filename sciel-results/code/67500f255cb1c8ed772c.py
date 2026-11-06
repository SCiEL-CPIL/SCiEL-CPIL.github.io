"""
Myerson-optimal finite auction for one additive bidder, two goods.
Values in {0,1,2,3} for each good. Three fixed distributions.
"""
import numpy as np
from scipy.optimize import linprog
from itertools import product

# Precompute the support: all (v1, v2) pairs where v1, v2 in {0,1,2,3}
# There are 16 possible value types.
# The allocation is a 2-vector (probability of allocating good 1, good 2).
# But since it's a single additive bidder, the "bundle" is determined by which goods are allocated.
# Actually, for a single additive bidder, the allocation rule maps each value type to a bundle (subset of goods).
# Since there are 2 goods, there are 4 possible bundles: {}, {1}, {2}, {1,2}.
# The allocation probability for each bundle given a value type must sum to 1 (feasibility).
# Actually, the problem says allocations are (16,2), so it's the probability of allocating each good.
# For a single additive bidder, incentive compatibility requires that the allocation is monotone in values.

# Let's think about this more carefully.
# The bidder has value v = (v1, v2) where v1, v2 in {0,1,2,3}.
# The mechanism assigns allocation a(v) = (a1(v), a2(v)) and payment p(v).
# Individual rationality: v·a(v) - p(v) >= 0 for all v.
# Incentive compatibility: v·a(v) - p(v) >= v'·a(v') - p(v') for all v, v'.
# 
# For a single additive bidder, the IC constraint is equivalent to:
# 1. The allocation is monotone: if v >= v' (component-wise), then a(v) >= a(v') (component-wise).
# 2. The payment is determined by the allocation via the envelope theorem:
#    p(v) = v·a(v) - U(v)
#    where U(v) is the utility, and U(v) = U(0) + integral along any path from 0 to v of a(w)·dw.
#    For IR, we need U(v) >= 0 for all v, so U(0) >= 0. To maximize revenue, set U(0) = 0 (or as low as possible).
#    Actually, U(0) must be >= 0 for IR at v=0. If we set U(0)=0, then U(v) must be >= 0 for all v.
#    U(v) = integral from 0 to v of a1(w1,w2) dw1 + a2(w1,w2) dw2 (along any path, by monotonicity).
#    For discrete values, U(v1,v2) = sum over paths. Since it's a lattice, we can use:
#    U(v1,v2) = sum_{i=1}^{v1} a1(i, v2) * 1 + sum_{j=1}^{v2} a2(v1, j) * 1
#    Wait, that's not quite right. The standard formula for discrete type spaces:
#    U(v) = U(0) + sum_{k: 0 < k <= v} a(k) · (k - k_prev)
#    Actually, for a lattice type space with unit increments, the utility difference is:
#    U(v1,v2) - U(v1-1,v2) = a1(v1,v2)
#    U(v1,v2) - U(v1,v2-1) = a2(v1,v2)
#    So U(v1,v2) = sum_{i=1}^{v1} a1(i, v2) + sum_{j=1}^{v2} a2(v1, j)
#    Let me verify: U(1,0) = a1(1,0). U(0,1) = a2(0,1). U(1,1) = a1(1,1) + a2(1,1)? 
#    No, that's not right either. Let's use the path from (0,0) to (v1,v2):
#    U(v1,v2) = sum_{i=1}^{v1} a1(i, v2) + sum_{j=1}^{v2} a2(v1, j)
#    Check: U(1,1) = a1(1,1) + a2(1,1). 
#    Alternative path: go to (1,0) then (1,1): U(1,0) + a2(1,1) = a1(1,0) + a2(1,1).
#    For IC, we need path independence, which requires a1(1,1) = a1(1,0) and a2(1,1) = a2(1,0)? No.
#    Actually, for the utility to be well-defined, we need the mixed partials to be equal, but since allocations are just functions, the IC condition is that the allocation is monotone, and then the utility is well-defined as the line integral, which is path-independent if and only if the allocation is monotone. Wait, no.
#    
#    Let me reconsider. For a single bidder with additive values on a finite lattice, the IC constraint is:
#    The allocation rule a: V -> R^2 must be monotone (non-decreasing in each component).
#    Then the payment is uniquely determined (up to a constant) by:
#    p(v) = v·a(v) - U(v)
#    where U(v) = sum_{i=1}^{v1} a1(i, v2) + sum_{j=1}^{v2} a2(v1, j)  [with U(0,0)=0]
#    And IR requires U(v) >= 0 for all v, and p(v) >= 0.
#
#    Actually, I think the correct formula is:
#    U(v1, v2) = U(0,0) + sum_{i=1}^{v1} a1(i, v2) + sum_{j=1}^{v2} a2(v1, j)
#    This is path-independent because:
#    Path 1: (0,0)->(1,0)->(1,1): a1(1,0) + a2(1,1)
#    Path 2: (0,0)->(0,1)->(1,1): a2(0,1) + a1(1,1)
#    For these to be equal, we need a1(1,0) + a2(1,1) = a2(0,1) + a1(1,1), i.e., a1(1,0) - a1(1,1) = a2(0,1) - a2(1,1).
#    This is NOT automatically satisfied by monotonicity. So my formula is wrong.
#
#    The correct approach: The IC constraint for a single bidder is that the allocation is monotone, and the payment satisfies the "no overpayment" condition. The utility is:
#    U(v) = v·a(v) - p(v)
#    IC: U(v) >= v·a(v') - p(v') for all v, v'.
#    This is equivalent to: the function U(v) = v·a(v) - p(v) is supermodular and a is monotone.
#    For a finite type space, we can solve for p given a by requiring that the IC constraints are tight along a "canonical path".
#
#    Actually, for a single additive bidder on a product space, the standard result is:
#    If a is monotone, then the minimum payment (maximizing revenue) consistent with IC and IR is:
#    p(v) = v·a(v) - U(v)
#    where U(v) = sum_{w: 0 <= w <= v} [a(w) - a(w-)]  where w- is w with one component decremented... 
#    
#    Let me just use the LP approach to be safe. I'll formulate the mechanism design as an LP.

def _get_distribution(distribution_spec):
    """Return the probability vector over the 16 value types."""
    # Value types are indexed by (v1, v2) with v1, v2 in {0,1,2,3}
    # Index = v1 * 4 + v2
    if distribution_spec == "uniform":
        # Uniform over {0,1,2,3}^2
        probs = np.ones(16) / 16.0
    elif distribution_spec == "independent_bernoulli":
        # Each good independently 0 or 1 with prob 0.5
        probs = np.zeros(16)
        for v1 in range(4):
            for v2 in range(4):
                if v1 in (0, 1) and v2 in (0, 1):
                    p1 = 0.5 if v1 == 1 else 0.5
                    p2 = 0.5 if v2 == 1 else 0.5
                    probs[v1 * 4 + v2] = p1 * p2
                else:
                    probs[v1 * 4 + v2] = 0.0
        # Normalize (should already be 1)
        probs = probs / probs.sum()
    elif distribution_spec == "correlated":
        # Correlated: (0,0) with 0.2, (1,1) with 0.3, (2,2) with 0.3, (3,3) with 0.2
        probs = np.zeros(16)
        probs[0 * 4 + 0] = 0.2
        probs[1 * 4 + 1] = 0.3
        probs[2 * 4 + 2] = 0.3
        probs[3 * 4 + 3] = 0.2
        probs = probs / probs.sum()
    else:
        # Default to uniform
        probs = np.ones(16) / 16.0
    return probs

def _solve_optimal_auction(distribution_spec, support, constraints):
    """
    Solve the optimal auction LP for a single additive bidder with two goods.
    Values in {0,1,2,3} for each good.
    """
    probs = _get_distribution(distribution_spec)
    
    # Value types: (v1, v2) for v1, v2 in 0..3
    # Index i = v1 * 4 + v2
    value_types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    values = np.array([[v1, v2] for v1, v2 in value_types], dtype=float)
    
    # Variables:
    # a[i, j] for i in 0..15, j in 0..1: allocation probability for good j given type i
    # p[i] for i in 0..15: payment for type i
    # Total variables: 16*2 + 16 = 48
    
    # Objective: maximize expected revenue = sum_i probs[i] * p[i]
    # Minimize: -sum_i probs[i] * p[i]
    
    c = np.zeros(48)
    for i in range(16):
        c[32 + i] = -probs[i]  # minimize -revenue
    
    # Constraints:
    # 1. Feasibility: 0 <= a[i,j] <= 1 for all i, j
    #    a[i,j] >= 0: -a[i,j] <= 0
    #    a[i,j] <= 1: a[i,j] <= 1
    # 2. Payments non-negative: p[i] >= 0
    # 3. Individual rationality: values[i]·a[i] - p[i] >= 0  =>  p[i] <= values[i]·a[i]
    # 4. Incentive compatibility: for all i, j:
    #    values[i]·a[i] - p[i] >= values[i]·a[j] - p[j]
    #    => p[i] - p[j] <= values[i]·a[i] - values[i]·a[j]
    
    # Build constraint matrix A_ub such that A_ub @ x <= b_ub
    # x = [a[0,0], a[0,1], ..., a[15,0], a[15,1], p[0], ..., p[15]]
    
    A_ub = []
    b_ub = []
    
    # Feasibility: a[i,j] >= 0 is handled by bounds. a[i,j] <= 1:
    for i in range(16):
        for j in range(2):
            row = np.zeros(48)
            row[i * 2 + j] = 1.0
            A_ub.append(row)
            b_ub.append(1.0)
    
    # IR: p[i] <= values[i]·a[i]
    # p[i] - values[i,0]*a[i,0] - values[i,1]*a[i,1] <= 0
    for i in range(16):
        row = np.zeros(48)
        row[32 + i] = 1.0
        row[i * 2 + 0] = -values[i, 0]
        row[i * 2 + 1] = -values[i, 1]
        A_ub.append(row)
        b_ub.append(0.0)
    
    # IC: for all i, j (i != j):
    # p[i] - p[j] <= values[i]·a[i] - values[i]·a[j]
    # p[i] - p[j] - values[i,0]*a[i,0] - values[i,1]*a[i,1] + values[i,0]*a[j,0] + values[i,1]*a[j,1] <= 0
    for i in range(16):
        for j in range(16):
            if i == j:
                continue
            row = np.zeros(48)
            row[32 + i] = 1.0
            row[32 + j] = -1.0
            row[i * 2 + 0] = -values[i, 0]
            row[i * 2 + 1] = -values[i, 1]
            row[j * 2 + 0] = values[i, 0]
            row[j * 2 + 1] = values[i, 1]
            A_ub.append(row)
            b_ub.append(0.0)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: 0 <= a[i,j] <= 1, 0 <= p[i] <= inf
    bounds = []
    for i in range(16):
        for j in range(2):
            bounds.append((0.0, 1.0))
    for i in range(16):
        bounds.append((0.0, None))
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback: no allocation, no payment
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return allocations, payments
    
    x = res.x
    allocations = x[:32].reshape(16, 2)
    payments = x[32:48]
    
    # Clip to ensure numerical feasibility
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.clip(payments, 0.0, None)
    
    return allocations, payments


def construct_auction(distribution_spec, support, constraints):
    """
    Construct the optimal auction mechanism.
    
    Args:
        distribution_spec: string specifying the distribution
        support: the value support
        constraints: additional constraints
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    allocations, payments = _solve_optimal_auction(distribution_spec, support, constraints)
    
    return {
        'allocations': allocations,
        'payments': payments
    }
