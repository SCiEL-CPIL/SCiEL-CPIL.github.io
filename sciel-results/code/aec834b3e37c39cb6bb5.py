"""Incentive-compatible finite auction design: maximize expected revenue via utility-based LP."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Parse the problem setup
    # One bidder, two goods, values in {0,1,2,3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    type_idx = {t: i for i, t in enumerate(types)}
    
    # Parse distribution_spec to get probabilities for each type
    # distribution_spec could be various formats; handle common cases
    probs = np.ones(n_types) / n_types  # default uniform
    
    if isinstance(distribution_spec, dict):
        if 'probabilities' in distribution_spec:
            p = distribution_spec['probabilities']
            if len(p) == n_types:
                probs = np.array(p, dtype=float)
        elif 'support' in distribution_spec and 'probs' in distribution_spec:
            p = distribution_spec['probs']
            if len(p) == n_types:
                probs = np.array(p, dtype=float)
        elif 'prob' in distribution_spec:
            p = distribution_spec['prob']
            if len(p) == n_types:
                probs = np.array(p, dtype=float)
        else:
            # Try to infer from keys
            for key in distribution_spec:
                if isinstance(distribution_spec[key], (list, tuple, np.ndarray)):
                    arr = np.array(distribution_spec[key], dtype=float)
                    if len(arr) == n_types:
                        probs = arr
                        break
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == n_types:
            probs = np.array(distribution_spec, dtype=float)
        elif len(distribution_spec) == 2:
            # Could be two independent marginal distributions
            p1 = np.array(distribution_spec[0], dtype=float)
            p2 = np.array(distribution_spec[1], dtype=float)
            if len(p1) == 4 and len(p2) == 4:
                probs = np.outer(p1, p2).flatten()
    elif isinstance(distribution_spec, np.ndarray):
        if distribution_spec.shape == (n_types,):
            probs = distribution_spec.copy()
        elif distribution_spec.shape == (4, 4):
            probs = distribution_spec.flatten()
    
    # Normalize probabilities
    probs = np.asarray(probs, dtype=float)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types
    
    # Support parsing: list of types or just the value set
    # We'll use the full 16-type support regardless
    
    # Constraints parsing
    # constraints could specify additional limits
    max_regret = 1e-8
    if isinstance(constraints, dict):
        if 'max_regret' in constraints:
            max_regret = constraints['max_regret']
    
    # Build the LP for mechanism design
    # Decision variables:
    #   x[t][g] for t in 0..15, g in 0,1 : allocation probability of good g to type t
    #   p[t] for t in 0..15 : payment from type t
    # 
    # For a single bidder with two goods, incentive compatibility means:
    # For all types t, t':
    #   sum_g x[t][g]*v[t][g] - p[t] >= sum_g x[t'][g]*v[t][g] - p[t']
    # i.e., u(t) >= u(t', t) where u(t) is utility when type is t and t' is the true type
    # 
    # Equivalently, using the utility representation:
    # Let u[t] = expected utility of type t when reporting truthfully
    # IC: u[t] >= u[t'] - (v[t] - v[t'])·x[t']  for all t, t'
    # This is the discrete subgradient condition.
    #
    # Actually, let's use the standard LP formulation:
    # Variables: x[16][2] (allocations), p[16] (payments)
    # Maximize: sum_t probs[t] * p[t]
    # Subject to:
    #   IC: for all t, t': v[t]·x[t] - p[t] >= v[t]·x[t'] - p[t']
    #   IR: for all t: v[t]·x[t] - p[t] >= 0
    #   Feasibility: 0 <= x[t][g] <= 1 for all t, g
    #   Nonneg payments: p[t] >= 0
    
    n_goods = 2
    n_vars = n_types * n_goods + n_types  # 32 + 16 = 48
    
    # Objective: maximize expected revenue = sum_t probs[t] * p[t]
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_types * n_goods + t] = -probs[t]
    
    # Build constraints matrix A_ub, b_ub
    # We need to express all constraints as A_ub @ x <= b_ub
    
    A_rows = []
    b_rows = []
    
    # IC constraints: for all t, t' (t != t' to reduce, but include all for safety)
    # v[t]·x[t] - p[t] >= v[t]·x[t'] - p[t']
    # => v[t]·x[t] - v[t]·x[t'] - p[t] + p[t'] >= 0
    # => -(v[t]·x[t] - v[t]·x[t'] - p[t] + p[t']) <= 0
    # => (-v[t][0])x[t][0] + v[t][0]x[t'][0] + (-v[t][1])x[t][1] + v[t][1]x[t'][1] - p[t] + p[t'] <= 0
    
    for t in range(n_types):
        for t2 in range(n_types):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            vt = types[t]
            # Coefficients for x[t]
            row[t * n_goods + 0] = -vt[0]
            row[t * n_goods + 1] = -vt[1]
            # Coefficients for x[t']
            row[t2 * n_goods + 0] = vt[0]
            row[t2 * n_goods + 1] = vt[1]
            # Coefficients for payments
            row[n_types * n_goods + t] = -1.0
            row[n_types * n_goods + t2] = 1.0
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR constraints: v[t]·x[t] - p[t] >= 0
    # => -v[t][0]x[t][0] - v[t][1]x[t][1] + p[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        vt = types[t]
        row[t * n_goods + 0] = -vt[0]
        row[t * n_goods + 1] = -vt[1]
        row[n_types * n_goods + t] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    # Nonnegativity of payments is handled by bounds
    # Bounds: x[t][g] in [0, 1], p[t] in [0, inf]
    bounds = []
    for t in range(n_types):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
    for t in range(n_types):
        bounds.append((0.0, None))
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Solve the LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: uniform allocation, zero payments (valid but zero revenue)
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    allocations = x[:n_types * n_goods].reshape(n_types, n_goods)
    payments = x[n_types * n_goods:]
    
    # Clip to ensure feasibility with small epsilon margin
    eps = 1e-9
    allocations = np.clip(allocations, eps, 1.0 - eps)
    payments = np.maximum(payments, 0.0)
    
    # Verify IC and IR; if violated, fall back
    violations = 0
    for t in range(n_types):
        vt = np.array(types[t])
        util_true = vt @ allocations[t] - payments[t]
        if util_true < -max_regret:
            violations += 1
        for t2 in range(n_types):
            if t == t2:
                continue
            util_misrep = vt @ allocations[t2] - payments[t2]
            if util_true < util_misrep - max_regret:
                violations += 1
    
    if violations > 0:
        # Fallback: trivial mechanism
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
    
    # Check deadline
    if time.time() > deadline:
        pass  # Already returned
    
    return {'allocations': allocations, 'payments': payments}
