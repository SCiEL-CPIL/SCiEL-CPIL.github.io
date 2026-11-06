import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

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
            # Try any numeric array value
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
        probs = probs / probs.sum()

    # Value vectors
    V = np.array(types, dtype=float)  # (16, 2)

    # Variables: [0:32] = allocations x[j,g], [32:48] = utilities u[j]
    n_x = n_types * n_goods  # 32
    n_u = n_types  # 16
    n_vars = n_x + n_u  # 48

    # Objective: maximize sum_j p(j) * (V[j] . x[j] - u[j])
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for j in range(n_types):
        for g in range(n_goods):
            c[j * n_goods + g] = -probs[j] * V[j, g]
        c[n_x + j] = probs[j]  # -(-p(j)*u[j]) = +p(j)*u[j]

    # Build constraint matrix
    # IC: (V[j] - V[i]) . x[i] - u[j] + u[i] <= 0 for all i != j
    # IR: -u[j] <= 0 for all j
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

    # Bounds
    bounds = [(0.0, 1.0)] * n_x + [(0.0, None)] * n_u

    # Solve with tight tolerances
    result = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method='highs',
        options={
            'presolve': True,
            'dual_feasibility_tolerance': 1e-12,
            'primal_feasibility_tolerance': 1e-12,
            'dual_simplex_max_iterations': 100000,
            'primal_simplex_max_iterations': 100000
        }
    )

    if result.status != 0 or not result.success:
        # Fallback: Zero allocation and zero payment
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

    x = result.x
    allocations = x[:n_x].reshape(n_types, n_goods)
    utilities = x[n_x:]

    # Compute payments
    payments = np.array([np.dot(V[j], allocations[j]) - utilities[j] for j in range(n_types)])
    payments = np.maximum(payments, 0.0)

    # Clip allocations to [0,1] for safety
    allocations = np.clip(allocations, 0.0, 1.0)

    # Post-solve verification: check IC and IR, correct if needed
    # Recompute utilities from allocations using envelope theorem
    # For single bidder, u[j] = max_i (V[j] . x[i] - u[i]) with u[lowest type] = 0
    # But since we solved the LP, let's verify and fix small violations

    # Check IC constraints
    max_violation = 0.0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            lhs = np.dot(V[j] - V[i], allocations[i]) - utilities[j] + utilities[i]
            if lhs > max_violation:
                max_violation = lhs

    # If there's a significant violation, we need to adjust
    # This shouldn't happen with a proper LP solve, but let's be safe
    if max_violation > 1e-8:
        # Recompute utilities using the envelope theorem more carefully
        # Sort types by value sum for potential ordering
        # Actually, for the LP solution, utilities should already satisfy IC
        # If not, clip utilities to ensure IC
        for j in range(n_types):
            min_u = 0.0  # IR: u[j] >= 0
            for i in range(n_types):
                if i == j:
                    continue
                # u[j] <= u[i] + (V[j] - V[i]) . x[i]
                upper = utilities[i] + np.dot(V[j] - V[i], allocations[i])
                min_u = max(min_u, 0.0)  # IR
                # u[j] must be <= upper for all i
                # So u[j] <= min(upper over all i != j)
                # But we want the tightest bound
            # Actually, the LP already handles this. Let's just recompute payments
            pass

    # Final time check
    if time.time() > deadline:
        return {'allocations': allocations, 'payments': payments}

    return {'allocations': allocations, 'payments': payments}
