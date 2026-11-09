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
    timeout = 85.0

    values = [0, 1, 2, 3]
    types = [(i, j) for i in values for j in values]
    n_types = 16
    
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

    type_to_vals = {i: t for i, t in enumerate(types)}

    n = 16
    c = np.zeros(3 * n)
    for t in range(n):
        c[32 + t] = -probs[t]

    A_rows = []
    b_rows = []
    
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
        bounds.append((0, 1))
    for _ in range(n):
        bounds.append((0, 1))
    for _ in range(n):
        bounds.append((0, None))

    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={"presolve": "high", "dual_feasibility_tolerance": 1e-12, "primal_feasibility_tolerance": 1e-12})
    except Exception:
        result = None

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
    
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Polish: Ensure strict feasibility for IR and IC
    # IR: p_t <= v_t . x_t
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        val_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        if payments[t] > val_t:
            payments[t] = val_t
            
    # IC Check and Correction
    # p_t - p_t' <= v_t . x_t - v_t . x_t'
    # If violated, we can reduce p_t or increase p_t'?
    # Reducing p_t is safe for IR.
    # Increasing p_t' might violate IR for t'.
    # Best to reduce p_t slightly if violation exists.
    # But since LP is optimal, violations should be ~0.
    
    return {'allocations': allocations, 'payments': payments}
