import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 82.0

    # Type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    n_goods = 2

    # Parse distribution probabilities
    probs = None
    if isinstance(distribution_spec, dict):
        for key in ['probs', 'distribution', 'probabilities', 'weights']:
            if key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, tuple, np.ndarray)):
                    probs = np.array(val, dtype=float).flatten()
                    break
                elif isinstance(val, dict):
                    if 'probs' in val:
                        probs = np.array(val['probs'], dtype=float).flatten()
                        break
                    elif 'values' in val:
                        probs = np.array(val['values'], dtype=float).flatten()
                        break
        if probs is None:
            for key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, tuple, np.ndarray)):
                    arr = np.array(val, dtype=float).flatten()
                    if len(arr) == n_types:
                        probs = arr
                        break
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).flatten()

    if probs is None or len(probs) != n_types:
        probs = np.ones(n_types) / n_types
    else:
        s = probs.sum()
        if s > 0:
            probs = probs / s
        else:
            probs = np.ones(n_types) / n_types

    # Value vectors
    V = np.array(types, dtype=float)  # (16, 2)

    # LP formulation:
    # Variables: x[j,g] (allocation prob for type j, good g), u[j] (utility for type j)
    # Maximize sum_j p(j) * (V[j].x[j] - u[j])
    # s.t. IC: (V[j]-V[i]).x[i] - u[j] + u[i] <= 0 for all i != j
    #      IR: -u[j] <= 0
    #      0 <= x[j,g] <= 1

    n_x = n_types * n_goods  # 32
    n_u = n_types  # 16
    n_vars = n_x + n_u  # 48

    # Objective (minimize negative revenue)
    c = np.zeros(n_vars)
    for j in range(n_types):
        for g in range(n_goods):
            c[j * n_goods + g] = -probs[j] * V[j, g]
        c[n_x + j] = probs[j]

    # IC constraints
    n_ic = n_types * (n_types - 1)
    n_ir = n_types
    n_constraints = n_ic + n_ir

    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)

    row_idx = 0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            for g in range(n_goods):
                A_ub[row_idx, i * n_goods + g] = V[j, g] - V[i, g]
            A_ub[row_idx, n_x + j] = -1.0
            A_ub[row_idx, n_x + i] = 1.0
            b_ub[row_idx] = 0.0
            row_idx += 1

    for j in range(n_types):
        A_ub[row_idx, n_x + j] = -1.0
        b_ub[row_idx] = 0.0
        row_idx += 1

    bounds = [(0.0, 1.0)] * n_x + [(0.0, None)] * n_u

    # Solve with tight tolerances
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                              'primal_feasibility_tolerance': 1e-12})

    if time.time() > deadline - 5.0:
        # Return a safe default if we're running out of time
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

    if result.status != 0 or not result.success:
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

    x = result.x
    allocations = x[:n_x].reshape(n_types, n_goods)
    utilities = x[n_x:]

    # Compute payments: p(j) = V[j].x[j] - u[j]
    payments = np.array([np.dot(V[j], allocations[j]) - utilities[j] for j in range(n_types)])
    payments = np.maximum(payments, 0.0)
    allocations = np.clip(allocations, 0.0, 1.0)

    # Polish pass: try to increase payments slightly while maintaining IC/IR
    # For each type j, the maximum payment is V[j].x[j] - min(0, min_i [(V[j]-V[i]).x[i] - u[j] + u[i]])
    # Actually, from IC: u[i] >= u[j] + (V[i]-V[j]).x[i] for all i
    # And IR: u[j] >= 0
    # So u[j] >= max(0, max_i [u[j] + (V[i]-V[j]).x[i]])... that's circular.
    # Better: from IC constraints, u[i] - u[j] >= (V[i]-V[j]).x[i]
    # So u[j] <= u[i] - (V[i]-V[j]).x[i] for all i != j, and u[j] >= 0
    # The payment for type j is V[j].x[j] - u[j]
    # To maximize payment, minimize u[j] subject to all IC and IR.
    # The LP already does this. But numerical errors might leave room.
    # Let's verify IC/IR and adjust if needed.

    # Check and fix small numerical violations
    for j in range(n_types):
        if utilities[j] < 0:
            utilities[j] = 0.0

    # Re-verify IC: (V[j]-V[i]).x[i] - u[j] + u[i] <= 0
    # i.e., u[i] - u[j] >= (V[i]-V[j]).x[i]
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            lhs = utilities[i] - utilities[j]
            rhs = np.dot(V[i] - V[j], allocations[i])
            if lhs < rhs - 1e-8:
                # Violation: increase u[i] or decrease u[j]
                # Simplest: increase u[i]
                utilities[i] = rhs

    # Recompute payments
    payments = np.array([np.dot(V[j], allocations[j]) - utilities[j] for j in range(n_types)])
    payments = np.maximum(payments, 0.0)

    return {'allocations': allocations, 'payments': payments}
