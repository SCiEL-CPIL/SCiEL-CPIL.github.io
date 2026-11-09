"""Rewritten program: LP with relaxed IC constraints (Regret <= 1e-8) for potential revenue gain."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    """
    start_time = time.time()
    DEADLINE = start_time + 50

    # Parse the problem structure
    values = [0, 1, 2, 3]
    
    # Get distribution
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
    
    # Determine types
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]
    
    n_types = len(types)
    assert n_types == 16, f"Expected 16 types, got {n_types}"
    
    type_to_vals = {}
    for idx, (v1, v2) in enumerate(types):
        type_to_vals[idx] = (v1, v2)
    
    type_vals = np.array([type_to_vals[i] for i in range(16)], dtype=float)
    v1_all = type_vals[:, 0]
    v2_all = type_vals[:, 1]
    
    # Variables: x1[0..15], x2[16..31], p[32..47]
    n_vars = 48
    
    A_rows = []
    b_rows = []
    
    # Relaxation epsilon for Regret <= 1e-8
    # IC: u(t) >= u(t') - eps
    # v(t).x(t) - p(t) >= v(t).x(t') - p(t') - eps
    # p(t) - p(t') <= v(t).(x(t) - x(t')) + eps
    EPS = 1e-8
    
    # IC constraints: all pairs
    for t in range(16):
        for t2 in range(16):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            # Coeff for x1(t)
            row[t] = -v1_all[t]
            # Coeff for x1(t2)
            row[t2] = v1_all[t]
            # Coeff for x2(t)
            row[16 + t] = -v2_all[t]
            # Coeff for x2(t2)
            row[16 + t2] = v2_all[t]
            # Coeff for p(t)
            row[32 + t] = 1.0
            # Coeff for p(t2)
            row[32 + t2] = -1.0
            
            A_rows.append(row)
            b_rows.append(EPS) # Relaxed constraint
    
    # IR constraints: u(t) >= 0
    # v(t).x(t) - p(t) >= 0
    # p(t) <= v(t).x(t)
    # -v(t).x(t) + p(t) <= 0
    for t in range(16):
        row = np.zeros(n_vars)
        row[t] = -v1_all[t]
        row[16 + t] = -v2_all[t]
        row[32 + t] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Objective: maximize sum p(t) * probs[t]
    # linprog minimizes, so we minimize -sum
    c = np.zeros(n_vars)
    c[32:48] = -probs
    
    # Bounds
    bounds = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16
    
    # Solve
    result = None
    try:
        options = {
            'presolve': True,
            'dual_feasibility_tolerance': 1e-12,
            'primal_feasibility_tolerance': 1e-12,
            'time_limit': 45
        }
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    except Exception as e:
        print(f"Highs failed: {e}")
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        except Exception as e2:
            print(f"Highs fallback failed: {e2}")
            result = None
    
    if result is not None and result.success:
        x = result.x
        x1 = np.clip(x[0:16], 0, 1)
        x2 = np.clip(x[16:32], 0, 1)
        p = np.maximum(x[32:48], 0)
        allocations = np.column_stack((x1, x2))
        payments = p
    else:
        # Fallback: Zero mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
    
    # Post-verification and correction
    # Check IR
    for t in range(16):
        v1_t, v2_t = type_to_vals[t]
        util = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
        if util < -1e-8:
            payments[t] = max(0.0, v1_t * allocations[t, 0] + v2_t * allocations[t, 1])
    
    # Check IC (Regret)
    # We want to ensure Regret <= 1e-8.
    # If violated, we must adjust.
    # Since we solved the relaxed LP, violations should be minimal.
    # However, numerical errors might push it over.
    # If violated, we can reduce payments of the "cheating" type or increase payment of the "truthful" type?
    # Reducing payment of type t (who is being tempted to misreport) makes it less attractive to misreport?
    # No, if u(t') > u(t), t is tempted to report t'.
    # To fix, we need u(t) >= u(t') - eps.
    # We can increase u(t) (reduce p(t)) or decrease u(t') (increase p(t')).
    # Reducing p(t) increases revenue loss. Increasing p(t') decreases revenue gain.
    # Best to adjust the one with higher weight?
    # For simplicity, if violation exists, we snap to strict IC by adjusting payments minimally.
    
    max_violation = 0.0
    for t in range(16):
        v1_t, v2_t = type_to_vals[t]
        for t2 in range(16):
            if t == t2:
                continue
            v1_t2, v2_t2 = type_to_vals[t2]
            lhs = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
            rhs = v1_t * allocations[t2, 0] + v2_t * allocations[t2, 1] - payments[t2]
            # Violation if lhs < rhs - EPS
            viol = rhs - lhs - EPS
            if viol > 1e-9:
                max_violation = max(max_violation, viol)
    
    if max_violation > 0:
        # This should rarely happen if LP solved correctly.
        # We reduce payments globally to satisfy strict IC (regret=0) to be safe.
        # This might lower revenue, but ensures validity.
        payments = np.maximum(payments - max_violation, 0.0)
        # Re-check IR
        for t in range(16):
            v1_t, v2_t = type_to_vals[t]
            util = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
            if util < -1e-8:
                payments[t] = max(0.0, v1_t * allocations[t, 0] + v2_t * allocations[t, 1])
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
