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
    deadline = start_time + 75  # 75 second deadline with 15s safety margin
    
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
    
    # Extract values
    values = np.zeros((n_reports, 2))
    for i, rep in enumerate(support):
        values[i, 0] = rep[0]
        values[i, 1] = rep[1]
    
    # LP Setup
    n_vars = 3 * n_reports # 32 x + 16 p
    
    # Objective: Maximize sum probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_reports):
        c[32 + i] = -probs[i]
    
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
        
    try:
        # Use highs with tight tolerances
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                         options={'presolve': True, 'dual_feasibility_tolerance': 1e-9, 'primal_feasibility_tolerance': 1e-9})
        
        if result.status == 0:
            x = np.zeros((n_reports, 2))
            p = np.zeros(n_reports)
            for i in range(n_reports):
                x[i][0] = result.x[2*i]
                x[i][1] = result.x[2*i+1]
                p[i] = result.x[32+i]
            
            x = np.clip(x, 0, 1)
            p = np.clip(p, 0, None)
            
            # Verify IC and IR
            ic_violated = False
            for i in range(n_reports):
                max_pay = values[i][0]*x[i][0] + values[i][1]*x[i][1]
                if p[i] > max_pay + 1e-7:
                    p[i] = max_pay
                    ic_violated = True # Mark as needing check, though clipping helps IR
                    
            # Check IC
            for i in range(n_reports):
                for ip in range(n_reports):
                    if i == ip:
                        continue
                    u_ii = values[i][0]*x[i][0] + values[i][1]*x[i][1] - p[i]
                    u_iip = values[i][0]*x[ip][0] + values[i][1]*x[ip][1] - p[ip]
                    if u_ii < u_iip - 1e-7:
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
                # Fallback to Independent Optimal Mechanism
                return _fallback_independent_mechanism(values, probs)
        else:
            return _fallback_independent_mechanism(values, probs)
            
    except Exception as e:
        return _fallback_independent_mechanism(values, probs)

def _fallback_independent_mechanism(values, probs):
    """
    Fallback: Solve for optimal single-good mechanism for each good independently.
    For a single good with values v in {0,1,2,3}, the optimal IC/IR mechanism 
    for a general distribution can be approximated by a threshold rule.
    Here we use a fixed threshold rule that is optimal for uniform:
    Allocate if v >= 2, pay 2.
    This is IC and IR for any distribution where the support is {0,1,2,3}.
    """
    n_reports = len(values)
    x = np.zeros((n_reports, 2))
    p = np.zeros(n_reports)
    
    for i in range(n_reports):
        v1 = values[i][0]
        v2 = values[i][1]
        
        # Good 1
        if v1 >= 2:
            x[i][0] = 1.0
            p[i] += 2.0
        else:
            x[i][0] = 0.0
            
        # Good 2
        if v2 >= 2:
            x[i][1] = 1.0
            p[i] += 2.0
        else:
            x[i][1] = 0.0
            
    return {
        'allocations': x,
        'payments': p
    }
# EVOLVE-BLOCK-END
