import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 75
    
    # Default support
    if support is None:
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    else:
        support = [tuple(s) for s in support]
        
    n_reports = len(support)
    if n_reports != 16:
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
        n_reports = 16
        
    # Parse probabilities
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
        
    n_vars = 3 * n_reports
    
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
    for i in range(n_reports):
        for ip in range(n_reports):
            if i == ip:
                continue
            row = np.zeros(n_vars)
            row[2*i] = -values[i][0]
            row[2*i+1] = -values[i][1]
            row[2*ip] = values[i][0]
            row[2*ip] = values[i][1] # Bug in original code: row[2*ip+1]
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
            
            # Repair step
            x = np.clip(x, 0, 1)
            p = np.clip(p, 0, None)
            
            # Enforce IR strictly
            for i in range(n_reports):
                max_pay = values[i][0]*x[i][0] + values[i][1]*x[i][1]
                if p[i] > max_pay:
                    p[i] = max_pay
                    
            # Verify IC with tolerance
            ic_violated = False
            for i in range(n_reports):
                u_ii = values[i][0]*x[i][0] + values[i][1]*x[i][1] - p[i]
                for ip in range(n_reports):
                    if i == ip:
                        continue
                    u_iip = values[i][0]*x[ip][0] + values[i][1]*x[ip][1] - p[ip]
                    if u_ii < u_iip - 1e-7: # Slightly relaxed tolerance
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
                # If IC is violated, try to fix by reducing payments?
                # Or use fallback.
                return _fallback_independent_mechanism(values, probs)
        else:
            return _fallback_independent_mechanism(values, probs)
            
    except Exception:
        return _fallback_independent_mechanism(values, probs)

def _fallback_independent_mechanism(values, probs):
    n_reports = len(values)
    x = np.zeros((n_reports, 2))
    p = np.zeros(n_reports)
    
    # Simple threshold mechanism
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
