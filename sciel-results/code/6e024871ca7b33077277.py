"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline with 10s safety margin
    
    # 16 reports: (v1, v2) with v1, v2 in {0, 1, 2, 3}
    values = list(range(4))
    reports = [(v1, v2) for v1 in values for v2 in values]
    n = len(reports)  # 16
    report_index = {r: i for i, r in enumerate(reports)}
    
    # Get probabilities for each report from distribution_spec
    # distribution_spec could be a list of 16 probabilities or a dict
    if isinstance(distribution_spec, dict):
        probs = np.array([distribution_spec.get(r, 0.0) for r in reports])
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) == n:
            probs = probs
        else:
            # Maybe it's a 4x4 grid
            probs = np.array(distribution_spec).reshape(4, 4).flatten()
    else:
        probs = np.ones(n) / n
    
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n) / n
    
    # Variables: x1[0..15], x2[0..15], p[0..16..31]
    # Total 48 variables
    n_vars = 3 * n  # 48
    
    # Objective: maximize sum(p_i * q_i) -> minimize -sum(p_i * q_i)
    c = np.zeros(n_vars)
    for i in range(n):
        c[2 * n + i] = -probs[i]  # negative for maximization
    
    # A_ub x <= b_ub
    A_ub_rows = []
    b_ub = []
    
    # IC constraints: for each true report t and reported report r:
    # v1_t * x1_r + v2_t * x2_r - p_r <= v1_t * x1_t + v2_t * x2_t - p_t
    # Rearranged: v1_t * x1_r + v2_t * x2_r - p_r - v1_t * x1_t - v2_t * x2_t + p_t <= 0
    for t_idx, t_report in enumerate(reports):
        v1_t, v2_t = t_report
        for r_idx, r_report in enumerate(reports):
            v1_r, v2_r = r_report
            row = np.zeros(n_vars)
            # +v1_t * x1_r
            row[r_idx] += v1_t
            # +v2_t * x2_r
            row[n + r_idx] += v2_t
            # -p_r
            row[2 * n + r_idx] -= 1
            # -v1_t * x1_t
            row[t_idx] -= v1_t
            # -v2_t * x2_t
            row[n + t_idx] -= v2_t
            # +p_t
            row[2 * n + t_idx] += 1
            A_ub_rows.append(row)
            b_ub.append(0.0)
    
    # IR constraints: v1_i * x1_i + v2_i * x2_i - p_i >= 0
    # -> -v1_i * x1_i - v2_i * x2_i + p_i <= 0
    for i in range(n):
        v1_i, v2_i = reports[i]
        row = np.zeros(n_vars)
        row[i] = -v1_i
        row[n + i] = -v2_i
        row[2 * n + i] = 1
        A_ub_rows.append(row)
        b_ub.append(0.0)
    
    # Bounds: 0 <= x1, x2, p
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, None)] * n
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub)
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: return zero allocations and payments
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {"allocations": allocations, "payments": payments}
    
    x_opt = result.x
    x1 = x_opt[:n]
    x2 = x_opt[n:2*n]
    p = x_opt[2*n:3*n]
    
    # Clip small negatives from numerical errors
    x1 = np.clip(x1, 0, 1)
    x2 = np.clip(x2, 0, 1)
    p = np.clip(p, 0, None)
    
    allocations = np.column_stack([x1, x2])
    payments = p
    
    # Check deadline
    if time.time() > deadline:
        return {"allocations": allocations, "payments": payments}
    
    return {"allocations": allocations, "payments": payments}
# EVOLVE-BLOCK-END
