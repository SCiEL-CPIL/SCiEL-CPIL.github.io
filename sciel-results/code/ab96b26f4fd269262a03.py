"""Incentive-compatible finite auction design via dual LP + primal recovery."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog


def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction with one additive bidder, two goods,
    values in {0,1,2,3}, using the optimal mechanism derived from an LP.

    Returns dict with 'allocations' (16,2) and 'payments' (16,).
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # 1. Define Types and Bundles
    values = [0, 1, 2, 3]
    types = []
    for v1 in values:
        for v2 in values:
            types.append((v1, v2))
    n_types = 16
    n_bundles = 4
    bundles = [(0, 0), (1, 0), (0, 1), (1, 1)]

    # 2. Parse Distribution
    probs = np.zeros(16)
    try:
        if isinstance(distribution_spec, dict):
            for i, t in enumerate(types):
                if t in distribution_spec:
                    probs[i] = distribution_spec[t]
                elif str(t) in distribution_spec:
                    probs[i] = distribution_spec[str(t)]
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(16) / 16.0
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) == 16:
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    probs = np.ones(16) / 16.0
            else:
                probs = np.ones(16) / 16.0
        else:
            probs = np.ones(16) / 16.0
    except Exception:
        probs = np.ones(16) / 16.0

    probs = np.maximum(probs, 0)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0

    # 3. Setup Value Matrix
    # val[i, j] = value of bundle j for type i
    val = np.zeros((n_types, n_bundles))
    for i, (v1, v2) in enumerate(types):
        for j, (x1, x2) in enumerate(bundles):
            val[i, j] = v1 * x1 + v2 * x2

    # 4. Primal LP: maximize expected payment subject to IC, IR, feasibility
    # Variables: x[i,j] (64), p[i] (16) -> 80 vars
    n_vars = 80

    c = np.zeros(n_vars)
    for i in range(n_types):
        c[64 + i] = -probs[i]

    # Equality: sum_j x[i,j] = 1
    A_eq = np.zeros((n_types, n_vars))
    b_eq = np.ones(n_types)
    for i in range(n_types):
        for j in range(n_bundles):
            A_eq[i, i * n_bundles + j] = 1.0

    # Inequality: IR + IC
    n_ineq = n_types + n_types * (n_types - 1)
    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)

    row = 0
    # IR: p[i] - sum_j val[i,j]*x[i,j] <= 0
    for i in range(n_types):
        A_ub[row, 64 + i] = 1.0
        for j in range(n_bundles):
            A_ub[row, i * n_bundles + j] = -val[i, j]
        row += 1

    # IC: p[i] - p[j] + val[i].x[j] - val[i].x[i] <= 0
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            A_ub[row, 64 + i] = 1.0
            A_ub[row, 64 + j] = -1.0
            for k in range(n_bundles):
                A_ub[row, j * n_bundles + k] += val[i, k]
                A_ub[row, i * n_bundles + k] -= val[i, k]
            row += 1

    bounds = [(0, None)] * n_vars

    # 5. Solve LP
    best_allocations = None
    best_payments = None
    best_obj = -np.inf

    try:
        if time.time() - start_time < DEADLINE - 5.0:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                             bounds=bounds, method='highs')
            if result.status == 0:
                x_opt = result.x[:64].reshape(n_types, n_bundles)
                p_opt = result.x[64:80]

                x_opt = np.maximum(x_opt, 0)
                p_opt = np.maximum(p_opt, 0)

                # Normalize allocations
                for i in range(n_types):
                    s = x_opt[i].sum()
                    if s > 1e-9:
                        x_opt[i] /= s
                    else:
                        x_opt[i] = np.zeros(n_bundles)
                        x_opt[i, 0] = 1.0

                allocations = np.zeros((n_types, 2))
                for i in range(n_types):
                    for j, (x1, x2) in enumerate(bundles):
                        allocations[i, 0] += x1 * x_opt[i, j]
                        allocations[i, 1] += x2 * x_opt[i, j]

                # Verify
                max_violation = 0.0
                for i in range(n_types):
                    u_i = val[i] @ x_opt[i] - p_opt[i]
                    if u_i < -max_violation:
                        max_violation = -u_i
                    for j in range(n_types):
                        if i == j:
                            continue
                        u_i_xj = val[i] @ x_opt[j] - p_opt[j]
                        u_i_xi = val[i] @ x_opt[i] - p_opt[i]
                        if u_i_xj - u_i_xi > max_violation:
                            max_violation = u_i_xj - u_i_xi

                if max_violation <= 1e-8:
                    p_opt = np.clip(p_opt, 0, None)
                    best_allocations = allocations
                    best_payments = p_opt
                    best_obj = float(probs @ p_opt)
                else:
                    # Try to fix by projecting payments down
                    for i in range(n_types):
                        u_i = val[i] @ x_opt[i]
                        p_opt[i] = min(p_opt[i], max(u_i, 0.0))
                    max_violation = 0.0
                    for i in range(n_types):
                        u_i = val[i] @ x_opt[i] - p_opt[i]
                        if u_i < -max_violation:
                            max_violation = -u_i
                        for j in range(n_types):
                            if i == j:
                                continue
                            u_i_xj = val[i] @ x_opt[j] - p_opt[j]
                            u_i_xi = val[i] @ x_opt[i] - p_opt[i]
                            if u_i_xj - u_i_xi > max_violation:
                                max_violation = u_i_xj - u_i_xi
                    if max_violation <= 1e-8:
                        best_allocations = allocations
                        best_payments = p_opt
                        best_obj = float(probs @ p_opt)
    except Exception:
        pass

    if best_allocations is not None:
        return {
            'allocations': best_allocations,
            'payments': best_payments
        }

    # 6. Fallback: zero mechanism (always valid)
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
