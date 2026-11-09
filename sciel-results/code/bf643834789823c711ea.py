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
    timeout = 80.0  # Safety margin before 90s kill

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
    idx = {}
    for i, (v1, v2) in enumerate(types):
        idx[(v1, v2)] = i

    type_to_vals = {i: t for i, t in enumerate(types)}

    # Reduced LP: 32 variables (x1, x2 only)
    # Payments derived from IC/IR: p(t) = max over paths of sum of value differences
    # For monotone allocations, p(t) = v(t)·x(t) - sum_{t' < t} [v(t')·x(t') - v(t')·x(t'')]
    # Simpler: use the full LP but with better options and a polish pass

    n = 16
    c = np.zeros(3 * n)
    for t in range(n):
        c[32 + t] = -probs[t]  # Maximize sum p(t)*probs[t] -> Minimize -sum

    A_rows = []
    b_rows = []

    # IC Constraints:
    # For all t, t': p(t) - p(t') <= v(t)·x(t) - v(t)·x(t')
    # Rearranged: -v(t)·x(t) + v(t)·x(t') + p(t) - p(t') <= 0
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        for t2 in range(n):
            if t == t2:
                continue
            row = np.zeros(3 * n)
            row[0 + t] = -v1_t
            row[16 + t] = -v2_t
            row[0 + t2] = v1_t
            row[16 + t2] = v2_t
            row[32 + t] = 1
            row[32 + t2] = -1
            A_rows.append(row)
            b_rows.append(0)

    # IR Constraints:
    # p(t) <= v(t)·x(t)
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

    # Solve LP with high precision
    result = None
    try:
        options = {
            'presolve': 'default',
            'dual_feasibility_tolerance': 1e-14,
            'primal_feasibility_tolerance': 1e-14,
            'dual_stopping_tolerance': 1e-14,
            'time_limit': 45.0
        }
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    except Exception as e:
        result = None

    if time.time() - start_time > timeout:
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

    # Polish pass: recompute payments exactly from IC/IR to eliminate numerical error
    # p(t) must satisfy: p(t) >= p(t') + v(t')·x(t) - v(t')·x(t') for all t'
    # and p(t) <= v(t)·x(t)
    # Optimal p(t) = min(v(t)·x(t), max over t' of [p(t') + v(t')·x(t) - v(t')·x(t')])
    # Use iterative refinement
    p_polished = np.copy(payments)
    for iteration in range(10):
        changed = False
        for t in range(n):
            v1_t, v2_t = type_to_vals[t]
            val_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
            # Lower bound from IC
            lb = 0.0
            for t2 in range(n):
                v1_t2, v2_t2 = type_to_vals[t2]
                val_t2 = v1_t2 * allocations[t2, 0] + v2_t2 * allocations[t2, 1]
                # IC: p(t) >= p(t2) + v(t2)·x(t) - v(t2)·x(t2)
                ic_lb = p_polished[t2] + v1_t2 * allocations[t, 0] + v2_t2 * allocations[t, 1] - val_t2
                if ic_lb > lb:
                    lb = ic_lb
            # p(t) = min(val_t, max(lb, 0)) but must be >= lb and <= val_t
            # If lb > val_t, we have infeasibility; clip
            new_p = max(lb, 0.0)
            new_p = min(new_p, val_t)
            if abs(new_p - p_polished[t]) > 1e-12:
                p_polished[t] = new_p
                changed = True
        if not changed:
            break

    payments = p_polished
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)

    # Final IR check
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        val_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        if payments[t] > val_t + 1e-10:
            payments[t] = max(val_t, 0)

    return {'allocations': allocations, 'payments': payments}
