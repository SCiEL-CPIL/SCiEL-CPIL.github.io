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
    deadline = start_time + 75  # 75 second deadline with 15s safety margin
    
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
    
    # Solve the reduced LP over allocation variables x only
    # Variables: x[0..31] where x[i*2 + g] = allocation of good g at report i
    n_alloc = n_reports * 2  # 32
    n_vars = n_alloc
    
    # Objective: maximize expected revenue
    # p_i = v1_i * x_{i,0} + v2_i * x_{i,1} - S1_i - S2_i
    # S1_i = sum_{k=0}^{v1_i - 1} x_{(k, v2_i), 0}
    # S2_i = sum_{m=0}^{v2_i - 1} x_{(v1_i, m), 1}
    
    # c vector for linprog (minimize -objective)
    c = np.zeros(n_vars)
    
    # We need to express p_i in terms of x variables to form the objective.
    # Revenue = sum_i probs[i] * p_i
    # p_i = v1_i * x_{i,0} + v2_i * x_{i,1} - sum_{k < v1_i} x_{(k, v2_i), 0} - sum_{m < v2_i} x_{(v1_i, m), 1}
    
    # Coefficient of x_{j, g} in the objective:
    # It appears in p_i for various i.
    # Let's build the objective coefficients directly.
    
    obj_coeff = np.zeros(n_vars)
    
    for i in range(n_reports):
        v1_i, v2_i = values[i]
        p_i_coeff = np.zeros(n_vars)
        
        # Term: v1_i * x_{i,0}
        p_i_coeff[i * 2 + 0] += v1_i
        # Term: v2_i * x_{i,1}
        p_i_coeff[i * 2 + 1] += v2_i
        
        # Term: - sum_{k=0}^{v1_i - 1} x_{(k, v2_i), 0}
        for k in range(int(v1_i)):
            idx_k = k * 4 + int(v2_i)
            p_i_coeff[idx_k * 2 + 0] -= 1.0
            
        # Term: - sum_{m=0}^{v2_i - 1} x_{(v1_i, m), 1}
        for m in range(int(v2_i)):
            idx_m = int(v1_i) * 4 + m
            p_i_coeff[idx_m * 2 + 1] -= 1.0
            
        obj_coeff += probs[i] * p_i_coeff
        
    # Minimize -Revenue
    c = -obj_coeff
    
    # Constraints:
    # 1. Bounds: 0 <= x <= 1
    bounds = [(0, 1)] * n_vars
    
    # 2. Monotonicity
    # x_{i,0} <= x_{j,0} if v1_i <= v1_j (and v2_i == v2_j? No, monotonicity is component-wise for the function x_1(v1, v2) in v1)
    # For good 0: x_{(v1, v2), 0} is non-decreasing in v1.
    # For good 1: x_{(v1, v2), 1} is non-decreasing in v2.
    
    A_rows = []
    b_rows = []
    
    # Monotonicity for good 0 (index 0)
    for v2 in range(4):
        for v1 in range(3):  # compare v1 and v1+1
            i_curr = v1 * 4 + v2
            i_next = (v1 + 1) * 4 + v2
            # x_{i_curr, 0} <= x_{i_next, 0}
            # x_{i_curr, 0} - x_{i_next, 0} <= 0
            row = np.zeros(n_vars)
            row[i_curr * 2 + 0] = 1
            row[i_next * 2 + 0] = -1
            A_rows.append(row)
            b_rows.append(0)
            
    # Monotonicity for good 1 (index 1)
    for v1 in range(4):
        for v2 in range(3):  # compare v2 and v2+1
            i_curr = v1 * 4 + v2
            i_next = v1 * 4 + (v2 + 1)
            # x_{i_curr, 1} <= x_{i_next, 1}
            # x_{i_curr, 1} - x_{i_next, 1} <= 0
            row = np.zeros(n_vars)
            row[i_curr * 2 + 1] = 1
            row[i_next * 2 + 1] = -1
            A_rows.append(row)
            b_rows.append(0)
            
    A_ub = np.array(A_rows) if A_rows else None
    b_ub = np.array(b_rows) if b_rows else None
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: simple mechanism
        return _simple_fallback(probs, values, n_reports)
    
    # Extract solution
    x_sol = result.x[:n_alloc].reshape(n_reports, 2)
    x_sol = np.clip(x_sol, 0, 1)
    
    # Compute payments
    p_sol = np.zeros(n_reports)
    for i in range(n_reports):
        v1_i, v2_i = values[i]
        p_sol[i] = v1_i * x_sol[i, 0] + v2_i * x_sol[i, 1]
        for k in range(int(v1_i)):
            idx_k = k * 4 + int(v2_i)
            p_sol[i] -= x_sol[idx_k, 0]
        for m in range(int(v2_i)):
            idx_m = int(v1_i) * 4 + m
            p_sol[i] -= x_sol[idx_m, 1]
            
    p_sol = np.clip(p_sol, 0, None)
    
    # Verify constraints
    if not _verify_constraints(x_sol, p_sol, probs, values, n_reports, regret_bound, ir_required):
        # Try to fix or use fallback
        x_sol, p_sol = _fix_constraints(x_sol, p_sol, probs, values, n_reports, regret_bound, ir_required)
        if not _verify_constraints(x_sol, p_sol, probs, values, n_reports, regret_bound, ir_required):
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


def _verify_constraints(allocations, payments, probs, values, n_reports, regret_bound, ir_required):
    """Verify IC, IR, feasibility, and nonnegative payments."""
    if np.any(allocations < -1e-9) or np.any(allocations > 1 + 1e-9):
        return False
    if np.any(payments < -1e-9):
        return False
    
    if ir_required:
        for j in range(n_reports):
            v1_j, v2_j = values[j]
            utility = v1_j * allocations[j, 0] + v2_j * allocations[j, 1] - payments[j]
            if utility < -1e-9:
                return False
    
    for j in range(n_reports):
        v1_j, v2_j = values[j]
        u_true = v1_j * allocations[j, 0] + v2_j * allocations[j, 1] - payments[j]
        for i in range(n_reports):
            if i == j:
                continue
            u_misrep = v1_j * allocations[i, 0] + v2_j * allocations[i, 1] - payments[i]
            if u_true - u_misrep < -1e-9:
                return False
                
    return True


def _fix_constraints(allocations, payments, probs, values, n_reports, regret_bound, ir_required):
    """Attempt to fix small constraint violations."""
    allocations = np.clip(allocations, 0, 1)
    payments = np.clip(payments, 0, None)
    return allocations, payments


def _simple_fallback(probs, values, n_reports):
    """Simple fallback: always allocate both goods, charge 0."""
    allocations = np.ones((n_reports, 2))
    payments = np.zeros(n_reports)
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
