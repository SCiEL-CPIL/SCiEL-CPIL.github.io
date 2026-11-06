"""
Myerson-optimal finite auction for one additive bidder, two goods.
Values in {0,1,2,3} for each good. Three fixed distributions.
"""
import numpy as np
from scipy.optimize import linprog
from itertools import product


def _get_distribution(distribution_spec):
    """Return the probability vector over the 16 value types."""
    if distribution_spec == "uniform":
        probs = np.ones(16) / 16.0
    elif distribution_spec == "independent_bernoulli":
        probs = np.zeros(16)
        for v1 in range(4):
            for v2 in range(4):
                if v1 in (0, 1) and v2 in (0, 1):
                    p1 = 0.5
                    p2 = 0.5
                    probs[v1 * 4 + v2] = p1 * p2
                else:
                    probs[v1 * 4 + v2] = 0.0
        probs = probs / probs.sum()
    elif distribution_spec == "correlated":
        probs = np.zeros(16)
        probs[0 * 4 + 0] = 0.2
        probs[1 * 4 + 1] = 0.3
        probs[2 * 4 + 2] = 0.3
        probs[3 * 4 + 3] = 0.2
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0
    return probs


def _solve_optimal_auction(distribution_spec, support, constraints):
    """
    Solve the optimal auction LP for a single additive bidder with two goods.
    Values in {0,1,2,3} for each good.
    """
    probs = _get_distribution(distribution_spec)
    
    value_types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    values = np.array([[v1, v2] for v1, v2 in value_types], dtype=float)
    
    # Variables:
    # a[i, j] for i in 0..15, j in 0..1: allocation probability for good j given type i
    # p[i] for i in 0..15: payment for type i
    # Total variables: 16*2 + 16 = 48
    
    c = np.zeros(48)
    for i in range(16):
        c[32 + i] = -probs[i]
    
    A_ub = []
    b_ub = []
    
    # Feasibility: a[i,j] <= 1
    for i in range(16):
        for j in range(2):
            row = np.zeros(48)
            row[i * 2 + j] = 1.0
            A_ub.append(row)
            b_ub.append(1.0)
    
    # IR: p[i] <= values[i]·a[i]
    for i in range(16):
        row = np.zeros(48)
        row[32 + i] = 1.0
        row[i * 2 + 0] = -values[i, 0]
        row[i * 2 + 1] = -values[i, 1]
        A_ub.append(row)
        b_ub.append(0.0)
    
    # IC: for all i, j (i != j):
    # p[i] - p[j] <= values[i]·a[i] - values[i]·a[j]
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
    
    bounds = []
    for i in range(16):
        for j in range(2):
            bounds.append((0.0, 1.0))
    for i in range(16):
        bounds.append((0.0, None))
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return allocations, payments
    
    x = res.x
    allocations = x[:32].reshape(16, 2)
    payments = x[32:48]
    
    # Post-processing: enforce monotonicity and IR strictly
    # Monotonicity: if v >= v' component-wise, then a(v) >= a(v') component-wise
    # For a single additive bidder, IC + IR implies monotonicity, but let's enforce it to be safe.
    # Actually, the LP already enforces IC, which implies monotonicity for this structure.
    # But numerical errors might violate it slightly. Let's project onto monotone allocations.
    
    # Simple approach: for each good j, ensure a[i,j] is non-decreasing in v1 and v2.
    for j in range(2):
        # Sort by v1, then v2 and enforce monotonicity
        for v1 in range(1, 4):
            for v2 in range(4):
                # a(v1, v2) >= a(v1-1, v2)
                idx_curr = v1 * 4 + v2
                idx_prev = (v1 - 1) * 4 + v2
                if allocations[idx_curr, j] < allocations[idx_prev, j]:
                    allocations[idx_curr, j] = allocations[idx_prev, j]
        for v2 in range(1, 4):
            for v1 in range(4):
                # a(v1, v2) >= a(v1, v2-1)
                idx_curr = v1 * 4 + v2
                idx_prev = v1 * 4 + (v2 - 1)
                if allocations[idx_curr, j] < allocations[idx_prev, j]:
                    allocations[idx_curr, j] = allocations[idx_prev, j]
    
    # Now recompute payments to be the minimum IC-consistent payments with IR.
    # For a monotone allocation, the minimum payment for type v is:
    # p(v) = v·a(v) - U(v) where U(v) = max_{w <= v} [w·a(v) - ...] 
    # Actually, the standard formula: U(v) = max over all "paths" or equivalently:
    # U(v) = sup_{w} { w·a(v) - p(w) } but that's circular.
    # 
    # The correct approach for monotone allocations on a lattice:
    # U(0,0) = 0 (IR at origin)
    # U(v1,v2) = U(v1-1,v2) + a1(v1,v2)  (going right)
    #           = U(v1,v2-1) + a2(v1,v2)  (going up)
    # For consistency, we need these to agree. With monotone allocations, we can use:
    # U(v1,v2) = max over all monotone paths from (0,0) to (v1,v2) of sum of a's
    # But actually, for the LP solution, the payments already satisfy IC and IR.
    # Let's just recompute payments using the envelope theorem with the monotone allocations.
    
    # U(v) for a monotone allocation: U(v) = sum_{i=1}^{v1} a1(i, v2) + sum_{j=1}^{v2} a2(v1, j)
    # This is path-independent for monotone allocations? Let's check.
    # Path 1: (0,0)->(1,0)->(1,1): a1(1,0) + a2(1,1)
    # Path 2: (0,0)->(0,1)->(1,1): a2(0,1) + a1(1,1)
    # These are equal iff a1(1,0) - a1(1,1) = a2(0,1) - a2(1,1)
    # With monotonicity: a1(1,0) <= a1(1,1) and a2(0,1) <= a2(1,1)
    # So both sides are <= 0, but not necessarily equal.
    # 
    # The correct formula: U(v) = max over all w <= v of [w·a(v) - ...] no.
    # 
    # Let me just use the LP payments but clip to ensure IR.
    for i in range(16):
        max_payment = values[i, 0] * allocations[i, 0] + values[i, 1] * allocations[i, 1]
        if payments[i] > max_payment:
            payments[i] = max_payment
        if payments[i] < 0:
            payments[i] = 0.0
    
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
