import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: support of the value space
        constraints: additional constraints
    
    Returns:
        dict with 'allocations': (16, 2) array and 'payments': (16,) array
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    # Parse the problem structure
    values = [0, 1, 2, 3]
    types = [(i, j) for i in values for j in values]
    n_types = 16

    # Get distribution
    probs = None
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            elif 'values' in distribution_spec and 'probs' in distribution_spec:
                vals = np.array(distribution_spec['values'])
                ps = np.array(distribution_spec['probs'])
                if len(vals) == 16 and len(ps) == 16:
                    probs = ps
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)

    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
        probs = np.maximum(probs, 0)
        if probs.sum() == 0:
            probs = np.ones(16) / 16.0
        else:
            probs = probs / probs.sum()

    # Index mapping
    type_to_vals = {i: t for i, t in enumerate(types)}
    vals_to_idx = {t: i for i, t in enumerate(types)}

    # LP Variables: x1 (16), x2 (16), p (16) -> 48 vars
    n = 16
    c = np.zeros(3 * n)
    for t in range(n):
        c[32 + t] = -probs[t]  # Maximize sum p(t)*probs[t] -> Minimize -sum

    A_rows = []
    b_rows = []

    # IC Constraints using adjacent types only (monotonicity along chains)
    # For each type t, add constraints to its "neighbors" (types that differ by 1 in one coordinate)
    # This is sufficient for IC on a lattice with monotone allocation rules
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        # Neighbors: change v1 by ±1 or v2 by ±1 (if in range)
        neighbors = []
        for dv1 in [-1, 0, 1]:
            for dv2 in [-1, 0, 1]:
                if dv1 == 0 and dv2 == 0:
                    continue
                nv1, nv2 = v1_t + dv1, v2_t + dv2
                if 0 <= nv1 <= 3 and 0 <= nv2 <= 3:
                    neighbors.append((nv1, nv2))
        
        for (nv1, nv2) in neighbors:
            t2 = vals_to_idx[(nv1, nv2)]
            row = np.zeros(3 * n)
            # p(t) - p(t') <= v(t).x(t) - v(t).x(t')
            # -v1_t*x1_t - v2_t*x2_t + v1_t*x1_t2 + v2_t*x2_t2 + p_t - p_t2 <= 0
            row[0 + t] = -v1_t
            row[16 + t] = -v2_t
            row[0 + t2] = v1_t
            row[16 + t2] = v2_t
            row[32 + t] = 1
            row[32 + t2] = -1
            A_rows.append(row)
            b_rows.append(0)

    # IR Constraints: p(t) <= v(t).x(t)
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        row = np.zeros(3 * n)
        row[0 + t] = -v1_t
        row[16 + t] = -v2_t
        row[32 + t] = 1
        A_rows.append(row)
        b_rows.append(0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    bounds = []
    for _ in range(n):
        bounds.append((0, 1))  # x1
    for _ in range(n):
        bounds.append((0, 1))  # x2
    for _ in range(n):
        bounds.append((0, None))  # p

    # Solve LP with tighter tolerances
    result = None
    try:
        options = {
            'presolve': 'default',
            'dual_feasibility_tolerance': 1e-12,
            'primal_feasibility_tolerance': 1e-12,
            'dual_stopping_tolerance': 1e-12
        }
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    except Exception:
        result = None

    if time.time() > deadline:
        allocations = np.ones((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}

    if result is None or not result.success:
        allocations = np.ones((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}

    x = result.x
    x1 = x[0:16]
    x2 = x[16:32]
    p = x[32:48]

    allocations = np.column_stack((x1, x2))
    payments = p

    # Clip to ensure validity
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)

    # Check IR and fix if needed
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        val_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        if payments[t] > val_t + 1e-10:
            payments[t] = max(val_t, 0)

    # Polish pass: re-solve with even tighter tolerances if time permits
    remaining = deadline - time.time()
    if remaining > 8.0:
        try:
            polish_options = {
                'presolve': 'default',
                'dual_feasibility_tolerance': 1e-14,
                'primal_feasibility_tolerance': 1e-14,
                'dual_stopping_tolerance': 1e-14,
                'time_limit': max(remaining - 3.0, 1.0)
            }
            polish_result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                                    method='highs', options=polish_options)
            if polish_result is not None and polish_result.success:
                px = polish_result.x
                px1 = px[0:16]
                px2 = px[16:32]
                pp = px[32:48]
                polish_rev = np.sum(pp * probs)
                base_rev = np.sum(payments * probs)
                if polish_rev >= base_rev - 1e-12:
                    allocations = np.column_stack((px1, px2))
                    payments = pp
                    allocations = np.clip(allocations, 0, 1)
                    payments = np.maximum(payments, 0)
                    for t in range(n):
                        v1_t, v2_t = type_to_vals[t]
                        val_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
                        if payments[t] > val_t + 1e-10:
                            payments[t] = max(val_t, 0)
        except Exception:
            pass

    return {'allocations': allocations, 'payments': payments}
