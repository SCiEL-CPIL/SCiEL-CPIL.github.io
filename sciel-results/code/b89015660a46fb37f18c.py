"""Rewritten program for incentive-compatible finite auction design."""
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
    deadline = start_time + 85  # 85 second deadline with 5s safety margin

    # Parse inputs
    if support is None:
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    else:
        support = [tuple(s) for s in support]
    
    n_reports = len(support)
    if n_reports != 16:
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
        n_reports = 16

    # Extract probabilities from distribution_spec
    probs = np.ones(n_reports) / n_reports
    if distribution_spec is not None:
        if isinstance(distribution_spec, (list, tuple, np.ndarray)):
            dist_arr = np.array(distribution_spec, dtype=float).flatten()
            if len(dist_arr) == n_reports:
                probs = dist_arr
                prob_sum = probs.sum()
                if prob_sum > 0:
                    probs = probs / prob_sum
                else:
                    probs = np.ones(n_reports) / n_reports
            else:
                probs = np.ones(n_reports) / n_reports
        elif isinstance(distribution_spec, dict):
            for i, rep in enumerate(support):
                if rep in distribution_spec:
                    probs[i] = distribution_spec[rep]
                elif i in distribution_spec:
                    probs[i] = distribution_spec[i]
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                probs = np.ones(n_reports) / n_reports
    
    probs = np.clip(probs, 0, 1)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(n_reports) / n_reports

    # Extract values for each report
    values = np.zeros((n_reports, 2))
    for i, rep in enumerate(support):
        values[i, 0] = rep[0]
        values[i, 1] = rep[1]

    # Set up the LP
    # Variables: [x[0][0], x[0][1], x[1][0], x[1][1], ..., x[15][0], x[15][1], p[0], p[1], ..., p[15]]
    n_vars = 2 * n_reports + n_reports  # 32 + 16 = 48

    # Objective: maximize sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_reports):
        c[32 + i] = -probs[i]

    # Inequality constraints
    A_rows = []
    b_rows = []
    
    # IR: p[i] <= values[i] @ x[i]
    for i in range(n_reports):
        row = np.zeros(n_vars)
        row[2*i] = -values[i][0]
        row[2*i+1] = -values[i][1]
        row[32+i] = 1
        A_rows.append(row)
        b_rows.append(0)
    
    # IC: p[i] - p[i'] <= values[i] @ (x[i] - x[i'])
    for i in range(n_reports):
        for ip in range(n_reports):
            if i == ip:
                continue
            row = np.zeros(n_vars)
            row[2*i] = -values[i][0]
            row[2*i+1] = -values[i][1]
            row[2*ip] = values[i][0]
            row[2*ip+1] = values[i][1]
            row[32+i] = 1
            row[32+ip] = -1
            A_rows.append(row)
            b_rows.append(0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    bounds = []
    for i in range(n_reports):
        bounds.append((0, 1))
        bounds.append((0, 1))
    for i in range(n_reports):
        bounds.append((0, None))
    
    def _check_and_fix(x, p, tol=1e-8):
        """Check IC/IR and fix if needed. Returns (x, p, is_valid)."""
        x = np.clip(x, 0, 1)
        p = np.clip(p, 0, None)
        
        # Fix IR
        for i in range(n_reports):
            max_pay = values[i][0]*x[i][0] + values[i][1]*x[i][1]
            if p[i] > max_pay + tol:
                p[i] = max_pay
        
        # Check IC
        for i in range(n_reports):
            for ip in range(n_reports):
                if i == ip:
                    continue
                u_ii = values[i][0]*x[i][0] + values[i][1]*x[i][1] - p[i]
                u_iip = values[i][0]*x[ip][0] + values[i][1]*x[ip][1] - p[ip]
                if u_ii < u_iip - tol:
                    return x, p, False
        return x, p, True

    def _safe_fallback():
        """Return a safe constant-allocation mechanism."""
        x_safe = np.ones((n_reports, 2))
        min_val = min(values[i][0] + values[i][1] for i in range(n_reports))
        p_safe = np.full(n_reports, min_val)
        for i in range(n_reports):
            max_pay = values[i][0]*x_safe[i][0] + values[i][1]*x_safe[i][1]
            p_safe[i] = min(p_safe[i], max_pay)
        return x_safe, p_safe

    def _solve_lp(c_vec, A_ub_mat, b_ub_vec, bounds_list, time_limit=60, tight_tol=False):
        """Solve LP with given parameters."""
        opts = {'presolve': True, 'time_limit': time_limit}
        if tight_tol:
            opts['dual_feasibility_tolerance'] = 1e-12
            opts['primal_feasibility_tolerance'] = 1e-12
        else:
            opts['dual_feasibility_tolerance'] = 1e-10
            opts['primal_feasibility_tolerance'] = 1e-10
        return linprog(c_vec, A_ub=A_ub_mat, b_ub=b_ub_vec, bounds=bounds_list, 
                       method='highs', options=opts)

    def _polish_solution(x, p, tol=1e-10):
        """
        Polish phase: identify binding constraints and re-solve on restricted
        allocation vertices for higher precision.
        """
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:
            return x, p
        
        # Identify allocation vertices: x[i] is at a corner if both components are 0 or 1
        # More generally, identify which allocation patterns are "active" (used with non-negligible probability)
        active_indices = []
        for i in range(n_reports):
            # Check if x[i] is close to a vertex (0 or 1 for each component)
            x0 = x[i][0]
            x1 = x[i][1]
            is_vertex = (abs(x0) < 1e-6 or abs(x0 - 1) < 1e-6) and \
                        (abs(x1) < 1e-6 or abs(x1 - 1) < 1e-6)
            if is_vertex:
                active_indices.append(i)
        
        # If not enough vertices identified, use all reports
        if len(active_indices) < 4:
            active_indices = list(range(n_reports))
        
        # Build restricted LP on active indices only
        n_active = len(active_indices)
        n_active_vars = 2 * n_active + n_active
        
        # Map original indices to active indices
        idx_map = {}
        for new_i, orig_i in enumerate(active_indices):
            idx_map[orig_i] = new_i
        
        # Objective
        c_active = np.zeros(n_active_vars)
        for new_i, orig_i in enumerate(active_indices):
            c_active[2*n_active + new_i] = -probs[orig_i]
        
        # Constraints
        A_active = []
        b_active = []
        
        # IR constraints
        for new_i, orig_i in enumerate(active_indices):
            row = np.zeros(n_active_vars)
            row[2*new_i] = -values[orig_i][0]
            row[2*new_i+1] = -values[orig_i][1]
            row[2*n_active + new_i] = 1
            A_active.append(row)
            b_active.append(0)
        
        # IC constraints (only between active reports)
        for new_i, orig_i in enumerate(active_indices):
            for new_ip, orig_ip in enumerate(active_indices):
                if orig_i == orig_ip:
                    continue
                row = np.zeros(n_active_vars)
                row[2*new_i] = -values[orig_i][0]
                row[2*new_i+1] = -values[orig_i][1]
                row[2*new_ip] = values[orig_i][0]
                row[2*new_ip+1] = values[orig_i][1]
                row[2*n_active + new_i] = 1
                row[2*n_active + new_ip] = -1
                A_active.append(row)
                b_active.append(0)
        
        A_active = np.array(A_active)
        b_active = np.array(b_active)
        
        bounds_active = []
        for _ in range(n_active):
            bounds_active.append((0, 1))
            bounds_active.append((0, 1))
        for _ in range(n_active):
            bounds_active.append((0, None))
        
        try:
            result_active = _solve_lp(c_active, A_active, b_active, bounds_active, 
                                       time_limit=15, tight_tol=True)
            if result_active.status == 0:
                # Reconstruct full solution
                x_full = np.zeros((n_reports, 2))
                p_full = np.zeros(n_reports)
                for new_i, orig_i in enumerate(active_indices):
                    x_full[orig_i] = result_active.x[2*new_i:2*new_i+2]
                    p_full[orig_i] = result_active.x[2*n_active + new_i]
                
                # For inactive reports, keep original allocations and set payment to 0
                # Actually, we need to ensure IC holds for all pairs.
                # Let's just use the polished active solution and verify.
                x_check, p_check, valid = _check_and_fix(x_full, p_full, tol=1e-9)
                if valid:
                    rev_new = np.dot(probs, p_check)
                    rev_old = np.dot(probs, p)
                    if rev_new >= rev_old - 1e-6:
                        return x_check, p_check
        except Exception:
            pass
        
        return x, p

    best_x, best_p, best_valid = None, None, False
    best_revenue = -1

    try:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 10  # leave 10s for polish
        if remaining < 5:
            remaining = 5
        
        result = _solve_lp(c, A_ub, b_ub, bounds, time_limit=int(remaining))
        
        if result.status == 0:
            x = np.zeros((n_reports, 2))
            p = np.zeros(n_reports)
            for i in range(n_reports):
                x[i][0] = result.x[2*i]
                x[i][1] = result.x[2*i+1]
                p[i] = result.x[32+i]
            
            x, p, valid = _check_and_fix(x, p)
            if valid:
                best_x, best_p, best_valid = x, p, True
                best_revenue = np.dot(probs, p)
            else:
                # Try to repair by reducing payments slightly
                for scale in [0.99, 0.98, 0.95, 0.9]:
                    p_try = p * scale
                    x_try, p_try, valid = _check_and_fix(x, p_try)
                    if valid:
                        best_x, best_p, best_valid = x_try, p_try, True
                        best_revenue = np.dot(probs, p_try)
                        break
        else:
            # LP failed
            x_safe, p_safe = _safe_fallback()
            best_x, best_p, best_valid = x_safe, p_safe, True
            best_revenue = np.dot(probs, p_safe)
    
    except Exception:
        x_safe, p_safe = _safe_fallback()
        best_x, best_p, best_valid = x_safe, p_safe, True
        best_revenue = np.dot(probs, p_safe)
    
    # Polish phase: try to improve precision
    if best_valid and best_x is not None:
        elapsed = time.time() - start_time
        if elapsed < deadline - 8:
            best_x, best_p = _polish_solution(best_x, best_p, tol=1e-10)
            # Re-verify
            best_x, best_p, valid = _check_and_fix(best_x, best_p, tol=1e-8)
            if not valid:
                # Revert to pre-polish
                # We need to recompute... let's just use the safe fallback if invalid
                x_safe, p_safe = _safe_fallback()
                best_x, best_p, best_valid = x_safe, p_safe, True

    if best_valid:
        return {
            'allocations': best_x,
            'payments': best_p
        }
    else:
        x_safe, p_safe = _safe_fallback()
        return {
            'allocations': x_safe,
            'payments': p_safe
        }
# EVOLVE-BLOCK-END
