"""Incentive-compatible finite auction design with LP-based optimal mechanism."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with one additive bidder, two goods.
    Values in {0,1,2,3} for each good, so 16 possible reports.
    Returns dict with allocations:(16,2) and payments:(16,)
    """
    start_time = time.time()
    deadline = start_time + 75  # 75 second deadline with 15s safety margin for evaluator overhead
    
    # The report space: (v1, v2) where v1, v2 in {0,1,2,3}
    # Index: i = v1 * 4 + v2
    values = np.array([(v1, v2) for v1 in range(4) for v2 in range(4)], dtype=float)
    n_reports = 16
    
    # Parse distribution_spec into a probability vector over 16 reports
    probs = _parse_distribution(distribution_spec, n_reports)
    
    # Parse constraints
    regret_bound = 1e-8
    ir_required = True
    if constraints is not None and isinstance(constraints, dict):
        regret_bound = constraints.get('regret_bound', 1e-8)
        ir_required = constraints.get('ir_required', True)
    
    # Check time before starting heavy computation
    if time.time() > deadline - 5:
        return _simple_fallback(probs, values, n_reports)

    # Solve the LP for optimal expected revenue
    # Variables: 
    # x[0..31]: allocation of good 0/1 for report 0..15 (x[i*2+g])
    # p[0..15]: payment for report 0..15
    # Total vars: 48
    
    n_alloc = n_reports * 2  # 32
    n_pay = n_reports  # 16
    n_vars = n_alloc + n_pay  # 48
    
    # Objective: maximize expected revenue = sum_i p_i * payment_i
    # Minimize: -sum_i p_i * payment_i
    c = np.zeros(n_vars)
    for i in range(n_reports):
        c[n_alloc + i] = -probs[i]
    
    # Inequality constraints: A_ub @ vars <= b_ub
    
    A_rows = []
    b_rows = []
    
    # IC constraints: For each true type j and reported type i (i != j)
    # U(j,j) >= U(i,j)
    # v1_j*x_j0 + v2_j*x_j1 - p_j >= v1_j*x_i0 + v2_j*x_i1 - p_i
    # v1_j*(x_j0 - x_i0) + v2_j*(x_j1 - x_i1) - p_j + p_i >= 0
    # -[v1_j*(x_j0 - x_i0) + v2_j*(x_j1 - x_i1) - p_j + p_i] <= 0
    # Coeffs for x_j0: -v1_j, x_j1: -v2_j
    # Coeffs for x_i0: v1_j, x_i1: v2_j
    # Coeffs for p_i: 1, p_j: -1
    
    for j in range(n_reports):
        v1_j, v2_j = values[j]
        for i in range(n_reports):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[j * 2 + 0] = -v1_j
            row[j * 2 + 1] = -v2_j
            row[i * 2 + 0] = v1_j
            row[i * 2 + 1] = v2_j
            row[n_alloc + i] = 1
            row[n_alloc + j] = -1
            A_rows.append(row)
            b_rows.append(0)
            
    # IR constraints: U(j,j) >= 0
    # v1_j*x_j0 + v2_j*x_j1 - p_j >= 0
    # -[v1_j*x_j0 + v2_j*x_j1 - p_j] <= 0
    # Coeffs for x_j0: -v1_j, x_j1: -v2_j
    # Coeffs for p_j: 1
    
    if ir_required:
        for j in range(n_reports):
            v1_j, v2_j = values[j]
            row = np.zeros(n_vars)
            row[j * 2 + 0] = -v1_j
            row[j * 2 + 1] = -v2_j
            row[n_alloc + j] = 1
            A_rows.append(row)
            b_rows.append(0)
    
    if A_rows:
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
    else:
        A_ub = None
        b_ub = None
    
    # Bounds:
    # Allocation: [0, 1]
    # Payment: [0, inf)
    bounds = []
    for _ in range(n_alloc):
        bounds.append((0, 1))
    for _ in range(n_pay):
        bounds.append((0, None))
    
    # Solve LP
    # Using HiGHS which is fast and robust
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        return _simple_fallback(probs, values, n_reports)
    
    if not result.success:
        return _simple_fallback(probs, values, n_reports)
    
    # Extract solution
    x_sol = result.x[:n_alloc].reshape(n_reports, 2)
    p_sol = result.x[n_alloc:n_alloc + n_pay]
    
    # Clip small numerical errors
    x_sol = np.clip(x_sol, 0, 1)
    p_sol = np.clip(p_sol, 0, None)
    
    # Verify constraints with tight tolerance
    if not _verify_constraints(x_sol, p_sol, probs, values, n_reports, 1e-9, ir_required):
        # Attempt to fix
        x_sol, p_sol = _fix_constraints(x_sol, p_sol, probs, values, n_reports, ir_required)
        if not _verify_constraints(x_sol, p_sol, probs, values, n_reports, 1e-9, ir_required):
            # If still invalid, return a safe fallback
            return _simple_fallback(probs, values, n_reports)
            
    return {
        'allocations': x_sol,
        'payments': p_sol
    }

def _parse_distribution(distribution_spec, n_reports):
    """Parse distribution_spec into a probability vector of length n_reports."""
    if distribution_spec is None:
        return np.ones(n_reports) / n_reports
    
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.shape == (n_reports,):
            total = arr.sum()
            if total > 0:
                return arr / total
            return np.ones(n_reports) / n_reports
        elif len(arr) == n_reports:
            total = arr.sum()
            if total > 0:
                return arr / total
            return np.ones(n_reports) / n_reports
        elif arr.ndim == 2 and arr.shape[0] == n_reports:
            probs = np.zeros(n_reports)
            for row in arr:
                if len(row) >= 3:
                    v1, v2, p = int(row[0]), int(row[1]), float(row[2])
                    idx = v1 * 4 + v2
                    if 0 <= idx < n_reports:
                        probs[idx] = p
            total = probs.sum()
            if total > 0:
                return probs / total
            return np.ones(n_reports) / n_reports
    
    if isinstance(distribution_spec, dict):
        probs = np.zeros(n_reports)
        for key, val in distribution_spec.items():
            if isinstance(key, tuple) and len(key) == 2:
                v1, v2 = int(key[0]), int(key[1])
                idx = v1 * 4 + v2
                if 0 <= idx < n_reports:
                    probs[idx] = float(val)
            elif isinstance(key, int) and 0 <= key < n_reports:
                probs[key] = float(val)
        total = probs.sum()
        if total > 0:
            return probs / total
        return np.ones(n_reports) / n_reports
    
    if callable(distribution_spec):
        try:
            result = distribution_spec()
            return _parse_distribution(result, n_reports)
        except:
            pass
        try:
            result = distribution_spec(range(4), range(4))
            return _parse_distribution(result, n_reports)
        except:
            pass
    
    return np.ones(n_reports) / n_reports

def _verify_constraints(allocations, payments, probs, values, n_reports, tol, ir_required):
    """Verify IC, IR, feasibility, and nonnegative payments."""
    # Feasibility
    if np.any(allocations < -tol) or np.any(allocations > 1 + tol):
        return False
    
    # Nonnegative payments
    if np.any(payments < -tol):
        return False
    
    # IR
    if ir_required:
        for j in range(n_reports):
            v1_j, v2_j = values[j]
            utility = v1_j * allocations[j, 0] + v2_j * allocations[j, 1] - payments[j]
            if utility < -tol:
                return False
    
    # IC
    for j in range(n_reports):
        v1_j, v2_j = values[j]
        u_true = v1_j * allocations[j, 0] + v2_j * allocations[j, 1] - payments[j]
        for i in range(n_reports):
            if i == j:
                continue
            u_misrep = v1_j * allocations[i, 0] + v2_j * allocations[i, 1] - payments[i]
            if u_true - u_misrep < -tol:
                return False
    
    return True

def _fix_constraints(allocations, payments, probs, values, n_reports, ir_required):
    """Attempt to fix small constraint violations by adjusting payments slightly."""
    allocations = np.clip(allocations, 0, 1)
    payments = np.clip(payments, 0, None)
    
    # Simple heuristic: if IR is violated, reduce payment. If IC is violated, 
    # it's harder to fix locally without breaking IR. 
    # Since LP solutions are usually exact, this is a last resort.
    # We just return the clipped version.
    return allocations, payments

def _simple_fallback(probs, values, n_reports):
    """Simple fallback: always allocate both goods, charge 0. 
    This is IR (utility = value >= 0) and IC (utility = value, independent of report? 
    No, if allocation is constant, U(j,j) = v1_j + v2_j, U(i,j) = v1_j + v2_j. 
    So U(j,j) = U(i,j). IC holds. IR holds. Revenue = 0.
    """
    allocations = np.ones((n_reports, 2))
    payments = np.zeros(n_reports)
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
