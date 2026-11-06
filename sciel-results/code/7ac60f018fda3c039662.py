"""Incentive-compatible finite auction design: LP-based mechanism optimization."""
import time
import numpy as np
from scipy.optimize import linprog
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Parse the problem structure
    # support: list of (v1, v2) pairs, values in {0,1,2,3}
    # distribution_spec: probabilities for each type in support
    # We need to handle various input formats
    
    # Determine types and their order
    if support is not None and len(support) > 0:
        # support is a list of (v1, v2) tuples
        types = list(support)
    else:
        # Default: all combinations of values 0-3
        types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    
    n_types = len(types)
    
    # Create index mapping
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Get probabilities
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            probs = np.array([distribution_spec.get(t, 0.0) for t in types], dtype=float)
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
            if len(probs) == n_types:
                pass
            else:
                # Try to map by index
                probs = np.zeros(n_types)
                for i, p in enumerate(distribution_spec):
                    if i < n_types:
                        probs[i] = p
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(n_types) / n_types
    
    # LP variables:
    # For each type t and each good g: allocation x[t, g] (probability of allocating good g to bidder with type t)
    # For each type t: payment q[t]
    # Total variables: 2 * n_types (allocations) + n_types (payments) = 3 * n_types
    
    n_goods = 2
    n_vars = 2 * n_types + n_types  # 2*n_types allocations + n_types payments
    
    # Variable indexing:
    # alloc[t, g] at index 2*t + g
    # pay[t] at index 2*n_types + t
    
    def alloc_idx(t, g):
        return 2 * t + g
    
    def pay_idx(t):
        return 2 * n_types + t
    
    # Objective: maximize expected revenue = sum_t probs[t] * q[t]
    # In linprog, we minimize, so use -revenue
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[pay_idx(t)] = -probs[t]
    
    # Constraints:
    # 1. Feasibility: 0 <= x[t,g] <= 1 for all t, g
    # 2. Nonnegative payments: q[t] >= 0
    # 3. Individual rationality: v[t].x[t] - q[t] >= 0 for all t
    #    => q[t] <= v[t].x[t]
    # 4. Incentive compatibility (truthfulness):
    #    For all types t and t': u(t, t') <= u(t, t)
    #    where u(t, t') = v[t'].x[t] - q[t] (utility when type t' reports t)
    #    So: v[t'].x[t] - q[t] <= v[t].x[t] - q[t]
    #    => v[t'].x[t] <= v[t].x[t] for all t, t'
    #    
    #    Wait, that's not right. Let me reconsider.
    #    
    #    Utility of type t when reporting t': u(t, t') = v[t].x[t'] - q[t']
    #    Truthfulness: u(t, t) >= u(t, t') for all t, t'
    #    => v[t].x[t] - q[t] >= v[t].x[t'] - q[t']
    #    => q[t] - q[t'] <= v[t].x[t] - v[t].x[t']
    #    
    #    This is the standard IC constraint.
    
    # Build inequality constraints: A_ub @ x <= b_ub
    
    A_rows = []
    b_rows = []
    
    # IR constraints: q[t] - v[t].x[t] <= 0 for all t
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[pay_idx(t)] = 1.0
        for g in range(n_goods):
            row[alloc_idx(t, g)] = -types[t][g]
        A_rows.append(row)
        b_rows.append(0.0)
    
    # IC constraints: q[t] - q[t'] - v[t].x[t] + v[t].x[t'] <= 0 for all t, t'
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            row = np.zeros(n_vars)
            row[pay_idx(t)] = 1.0
            row[pay_idx(tp)] = -1.0
            for g in range(n_goods):
                row[alloc_idx(t, g)] = -types[t][g]
                row[alloc_idx(tp, g)] = types[t][g]
            A_rows.append(row)
            b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= x[t,g] <= 1, 0 <= q[t] <= inf
    bounds = []
    for t in range(n_types):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
    for t in range(n_types):
        bounds.append((0.0, None))
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: return uniform allocation with zero payments
        allocations = np.ones((n_types, n_goods)) / 2.0
        payments = np.zeros(n_types)
        return {"allocations": allocations, "payments": payments}
    
    x = result.x
    
    # Extract allocations and payments
    allocations = np.zeros((n_types, n_goods))
    for t in range(n_types):
        for g in range(n_goods):
            allocations[t, g] = x[alloc_idx(t, g)]
    
    payments = np.zeros(n_types)
    for t in range(n_types):
        payments[t] = x[pay_idx(t)]
    
    # Clip small numerical errors
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.clip(payments, 0.0, None)
    
    return {"allocations": allocations, "payments": payments}
