"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin within 60s limit
    
    # Parse the problem
    # support: likely a list or array of (v1, v2) tuples/arrays
    # distribution_spec: probabilities for each support point
    
    # Convert support to list of tuples to ensure hashability
    if isinstance(support, np.ndarray):
        support_list = [tuple(row) for row in support]
    elif isinstance(support, (list, tuple)):
        support_list = [tuple(v) for v in support]
    else:
        support_list = [tuple(support)]
        
    values = support_list
    n_values = len(values)
    
    # Reports are all combinations of {0,1,2,3} x {0,1,2,3}
    report_values = [(r1, r2) for r1 in range(4) for r2 in range(4)]
    n_reports = len(report_values)
    report_index = {r: i for i, r in enumerate(report_values)}
    
    # Get probabilities
    if isinstance(distribution_spec, dict):
        probs = np.array([distribution_spec.get(v, 0.0) for v in values])
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        # If length doesn't match, try to infer or default to uniform
        if len(probs) != n_values:
             # If distribution_spec is a single number, treat as uniform? Or maybe it's a param?
             # Assuming standard interface: list of probs matching support length.
             # If mismatch, fallback to uniform to avoid crash.
             if len(probs) == 1:
                 probs = np.full(n_values, probs[0])
             else:
                 probs = np.ones(n_values) / n_values
    else:
        probs = np.ones(n_values) / n_values
    
    # Normalize
    total_prob = probs.sum()
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(n_values) / n_values
    
    # Decision variables: x_{r,1}, x_{r,2} for each report r (allocation probs), p_r for each report r
    # Total variables: 2 * n_reports + n_reports = 3 * n_reports = 48
    # 
    # Objective: maximize expected revenue = sum_v prob_v * p_v (payment when bidder reports truthfully)
    # Wait, revenue is sum over true values of prob_v * payment when reporting truthfully
    # Actually, expected revenue = sum_v pi_v * p_v (payment when bidder reports true value)
    # 
    # But we need to maximize sum_v pi_v * p_v
    # 
    # Actually, let me reconsider. The expected revenue is:
    # E[revenue] = sum_v pi_v * p(v) where p(v) is the payment when the bidder with value v reports v
    # 
    # But wait, the mechanism is: for each report r, allocate x(r) and charge p(r).
    # The expected revenue is sum_v pi_v * p(v) (since truthful reporting is optimal)
    # 
    # Constraints:
    # 1. 0 <= x_{r,i} <= 1 for all r, i
    # 2. p_r >= 0 for all r
    # 3. IR: for all v: sum_i x_{v,i} * v_i - p_v >= 0
    # 4. IC: for all v, r: sum_i x_{v,i} * v_i - p_v >= sum_i x_{r,i} * v_i - p_r
    #    i.e., sum_i (x_{v,i} - x_{r,i}) * v_i >= p_v - p_r
    
    # Let's set up the LP
    # Variables: x_{r,1}, x_{r,2}, p_r for each report r
    # Index: for report r at index k, variables are at 3k, 3k+1, 3k+2
    
    n_vars = 3 * n_reports
    
    # Objective: maximize sum_v pi_v * p_v = maximize -sum_v pi_v * p_v (for linprog minimization)
    c = np.zeros(n_vars)
    for v_idx, v in enumerate(values):
        r_idx = report_index.get(v, None)
        if r_idx is not None:
            c[3 * r_idx + 2] = -probs[v_idx]  # minimize negative revenue
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_ub_rows = []
    b_ub_vals = []
    
    # IR constraints: for each v, sum_i x_{v,i} * v_i - p_v >= 0
    # => -sum_i x_{v,i} * v_i + p_v <= 0
    for v_idx, v in enumerate(values):
        r_idx = report_index.get(v, None)
        if r_idx is not None:
            row = np.zeros(n_vars)
            row[3 * r_idx] = -v[0]      # -x_{v,1} * v1
            row[3 * r_idx + 1] = -v[1]   # -x_{v,2} * v2
            row[3 * r_idx + 2] = 1.0     # +p_v
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # IC constraints: for all v, r: sum_i (x_{v,i} - x_{r,i}) * v_i >= p_v - p_r
    # => sum_i x_{v,i} * v_i - p_v - sum_i x_{r,i} * v_i + p_r <= 0
    for v_idx, v in enumerate(values):
        v_r_idx = report_index.get(v, None)
        if v_r_idx is None:
            continue
        for r in report_values:
            r_r_idx = report_index[r]
            row = np.zeros(n_vars)
            # sum_i x_{v,i} * v_i
            row[3 * v_r_idx] = v[0]
            row[3 * v_r_idx + 1] = v[1]
            # -p_v
            row[3 * v_r_idx + 2] = -1.0
            # -sum_i x_{r,i} * v_i
            row[3 * r_r_idx] -= v[0]
            row[3 * r_r_idx + 1] -= v[1]
            # +p_r
            row[3 * r_r_idx + 2] += 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # Bounds: x in [0,1], p >= 0
    bounds = []
    for k in range(n_reports):
        bounds.append((0, 1))  # x_{r,1}
        bounds.append((0, 1))  # x_{r,2}
        bounds.append((0, None))  # p_r
    
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if not result.success:
            raise ValueError(f"LP failed: {result.message}")
        
        x_opt = result.x
        # Extract allocations and payments
        allocations = np.zeros((n_reports, 2))
        payments = np.zeros(n_reports)
        for k in range(n_reports):
            allocations[k, 0] = x_opt[3 * k]
            allocations[k, 1] = x_opt[3 * k + 1]
            payments[k] = x_opt[3 * k + 2]
        
        # Compute expected revenue
        expected_revenue = 0.0
        for v_idx, v in enumerate(values):
            r_idx = report_index.get(v, None)
            if r_idx is not None:
                expected_revenue += probs[v_idx] * payments[r_idx]
        
        return {
            'allocations': allocations,
            'payments': payments
        }
    except Exception as e:
        # Fallback: simple threshold mechanism
        allocations = np.zeros((n_reports, 2))
        payments = np.zeros(n_reports)
        for k, r in enumerate(report_values):
            # Allocate good 1 if r1 >= 2, good 2 if r2 >= 2
            allocations[k, 0] = 1.0 if r[0] >= 2 else 0.0
            allocations[k, 1] = 1.0 if r[1] >= 2 else 0.0
            payments[k] = 2.0 * allocations[k, 0] + 2.0 * allocations[k, 1]
        
        return {
            'allocations': allocations,
            'payments': payments
        }
# EVOLVE-BLOCK-END
