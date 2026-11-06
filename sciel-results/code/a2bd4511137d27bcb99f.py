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
    deadline = start_time + 70  # 20s safety margin before 90s kill
    
    # Parse inputs
    if support is None:
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    else:
        support = [tuple(s) for s in support]
        
    n_reports = len(support)
    if n_reports != 16:
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
        n_reports = 16
        
    # Extract probabilities
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
        
    # Values
    values = np.zeros((n_reports, 2))
    for i, rep in enumerate(support):
        values[i, 0] = rep[0]
        values[i, 1] = rep[1]
        
    # Check if support is the standard grid
    is_standard_grid = True
    for i, rep in enumerate(support):
        v1, v2 = rep
        if not (0 <= v1 <= 3 and 0 <= v2 <= 3):
            is_standard_grid = False
            break
            
    # Strategy: Use Reduced LP with Transitivity for Standard Grid
    # For a single bidder, IC constraints imply concavity of utility u(t).
    # On a grid, if IC holds for adjacent points, it holds for all pairs.
    # This reduces constraints from ~240 (full IC + IR) to ~32 (adjacent IC + IR).
    
    n_vars = 3 * n_reports  # 2 allocations per report + 1 payment per report
    
    # Objective: Maximize sum probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_reports):
        c[2*n_reports + i] = -probs[i]
        
    A_rows = []
    b_rows = []
    
    # IR: p[i] <= values[i] @ x[i]  =>  -values[i]@x[i] + p[i] <= 0
    for i in range(n_reports):
        row = np.zeros(n_vars)
        row[2*i] = -values[i][0]
        row[2*i+1] = -values[i][1]
        row[2*n_reports+i] = 1
        A_rows.append(row)
        b_rows.append(0)
        
    # IC: u(i,i) >= u(i,ip)
    # v_i x_i - p_i >= v_i x_ip - p_ip
    # -v_i x_i + v_i x_ip + p_i - p_ip <= 0
    
    if is_standard_grid:
        # Generate adjacent pairs
        idx_to_v = {}
        for i, rep in enumerate(support):
            idx_to_v[rep] = i
            
        adj_pairs = []
        for v1 in range(4):
            for v2 in range(4):
                curr = (v1, v2)
                if curr not in idx_to_v: continue
                i = idx_to_v[curr]
                
                # Neighbor right: (v1, v2+1)
                if v2 < 3:
                    nxt = (v1, v2+1)
                    if nxt in idx_to_v:
                        j = idx_to_v[nxt]
                        adj_pairs.append((i, j))
                
                # Neighbor up: (v1+1, v2)
                if v1 < 3:
                    nxt = (v1+1, v2)
                    if nxt in idx_to_v:
                        j = idx_to_v[nxt]
                        adj_pairs.append((i, j))
        
        for i, ip in adj_pairs:
            row = np.zeros(n_vars)
            row[2*i] = -values[i][0]
            row[2*i+1] = -values[i][1]
            row[2*ip] = values[i][0]
            row[2*ip+1] = values[i][1]
            row[2*n_reports+i] = 1
            row[2*n_reports+ip] = -1
            A_rows.append(row)
            b_rows.append(0)
    else:
        # Full set of IC constraints for non-standard grids
        for i in range(n_reports):
            for ip in range(n_reports):
                if i == ip:
                    continue
                row = np.zeros(n_vars)
                row[2*i] = -values[i][0]
                row[2*i+1] = -values[i][1]
                row[2*ip] = values[i][0]
                row[2*ip+1] = values[i][1]
                row[2*n_reports+i] = 1
                row[2*n_reports+ip] = -1
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
        
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                         options={'presolve': True, 'dual_feasibility_tolerance': 1e-12, 'primal_feasibility_tolerance': 1e-12})
        
        if result.status == 0:
            x = np.zeros((n_reports, 2))
            p = np.zeros(n_reports)
            for i in range(n_reports):
                x[i][0] = result.x[2*i]
                x[i][1] = result.x[2*i+1]
                p[i] = result.x[2*n_reports+i]
            
            x = np.clip(x, 0, 1)
            p = np.clip(p, 0, None)
            
            # Verify constraints strictly
            ic_violated = False
            # IR Check
            for i in range(n_reports):
                max_pay = values[i][0]*x[i][0] + values[i][1]*x[i][1]
                if p[i] > max_pay + 1e-8:
                    p[i] = max_pay
                    ic_violated = True
                    
            # IC Check (Full verification to be safe, even if we used reduced constraints)
            for i in range(n_reports):
                for ip in range(n_reports):
                    if i == ip:
                        continue
                    u_ii = values[i][0]*x[i][0] + values[i][1]*x[i][1] - p[i]
                    u_iip = values[i][0]*x[ip][0] + values[i][1]*x[ip][1] - p[ip]
                    if u_ii < u_iip - 1e-8:
                        ic_violated = True
                        break
                if ic_violated:
                    break
            
            if not ic_violated:
                return {
                    'allocations': x,
                    'payments': p
                }
            else:
                return _fallback_independent_mechanism(values, probs)
        else:
            return _fallback_independent_mechanism(values, probs)
            
    except Exception:
        return _fallback_independent_mechanism(values, probs)

def _fallback_independent_mechanism(values, probs):
    n_reports = len(values)
    x = np.zeros((n_reports, 2))
    p = np.zeros(n_reports)
    
    # Fallback: Allocate good j if value_j >= 2, charge 2.
    # This is IC and IR.
    for i in range(n_reports):
        v1 = values[i][0]
        v2 = values[i][1]
        
        if v1 >= 2:
            x[i][0] = 1.0
            p[i] += 2.0
        if v2 >= 2:
            x[i][1] = 1.0
            p[i] += 2.0
            
    return {
        'allocations': x,
        'payments': p
    }
