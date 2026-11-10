"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog

# EVOLVE-BLOCK-START
def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with 1 additive bidder, 2 goods.
    Values in {0,1,2,3}. Three fixed distributions.
    Returns: {'allocations': (16,2) array of probabilities, 'payments': (16,) array}
    """
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    # Generate all 16 types
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = len(types)  # 16
    
    # Parse probabilities from distribution_spec
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'])
        elif 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if dist == 'uniform':
                probs = np.ones(n_types) / n_types
            elif dist == 'geometric':
                probs = np.array([0.7**v1 * 0.7**v2 for v1, v2 in types])
                probs /= probs.sum()
            else:
                probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec)
        if len(probs) == n_types:
            probs = probs / probs.sum()
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    # Ensure probabilities are valid
    probs = np.maximum(probs, 0)
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(n_types) / n_types
    
    # Variables:
    # x[i] = probability of allocating good 1 to type i (16 vars)
    # y[i] = probability of allocating good 2 to type i (16 vars)
    # p[i] = payment for type i (16 vars)
    # Total: 48 variables
    # Order: [x_0..x_15, y_0..y_15, p_0..p_15]
    
    n_vars = 3 * n_types  # 48
    
    # Objective: Maximize expected revenue = sum_i probs[i] * p[i]
    # c = [0,...,0, 0,...,0, probs[0],...,probs[15]]
    c = np.zeros(n_vars)
    c[2*n_types:3*n_types] = probs  # Maximize revenue (linprog minimizes, so use -c)
    c = -c  # Negate for minimization
    
    # Build inequality constraints A_ub @ vars <= b_ub
    A_rows = []
    b_rows = []
    
    # IR constraints: -v1_i * x[i] - v2_i * y[i] + p[i] <= 0
    for i in range(n_types):
        v1, v2 = types[i]
        row = np.zeros(n_vars)
        row[i] = -v1           # x[i]
        row[n_types + i] = -v2 # y[i]
        row[2*n_types + i] = 1 # p[i]
        A_rows.append(row)
        b_rows.append(0)
    
    # IC constraints: for each i (true type) and j (reported type)
    # (x[j]-x[i])*v1_i + (y[j]-y[i])*v2_i - p[j] + p[i] <= 0
    for i in range(n_types):
        v1_i, v2_i = types[i]
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[j] = v1_i
            row[i] = -v1_i
            row[n_types + j] = v2_i
            row[n_types + i] = -v2_i
            row[2*n_types + j] = -1
            row[2*n_types + i] = 1
            A_rows.append(row)
            b_rows.append(0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= x[i] <= 1, 0 <= y[i] <= 1, p[i] >= 0
    bounds = [(0, 1)] * n_types + [(0, 1)] * n_types + [(0, None)] * n_types
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: return a trivial feasible solution
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            return {'allocations': allocations, 'payments': payments}
        
        x = result.x[:n_types]
        y = result.x[n_types:2*n_types]
        p = result.x[2*n_types:3*n_types]
        
        # Clip to ensure feasibility
        x = np.clip(x, 0, 1)
        y = np.clip(y, 0, 1)
        p = np.maximum(p, 0)
        
        # Post-processing: tighten payments while maintaining feasibility
        # For each type, the maximum payment is min(v1*x[i] + v2*y[i], 
        #   min over j != i of (v1_i*x[j] + v2_i*y[j] - p[j] + (x[i]*v1_i + y[i]*v2_i)))
        # But we need to be careful not to violate IC
        
        # Verify and fix IR
        for i in range(n_types):
            v1, v2 = types[i]
            max_p_ir = x[i]*v1 + y[i]*v2
            if p[i] > max_p_ir + 1e-10:
                p[i] = max(0, max_p_ir)
        
        # Verify IC and fix if needed
        ic_violated = False
        for i in range(n_types):
            v1_i, v2_i = types[i]
            for j in range(n_types):
                if i == j:
                    continue
                util_report = x[j]*v1_i + y[j]*v2_i - p[j]
                util_truth = x[i]*v1_i + y[i]*v2_i - p[i]
                if util_report - util_truth > 1e-8:
                    ic_violated = True
                    break
            if ic_violated:
                break
        
        if ic_violated:
            # Try to fix by reducing payments slightly
            for i in range(n_types):
                v1, v2 = types[i]
                p[i] = max(0, min(p[i], x[i]*v1 + y[i]*v2))
            
            # Re-verify
            ic_violated = False
            for i in range(n_types):
                v1_i, v2_i = types[i]
                for j in range(n_types):
                    if i == j:
                        continue
                    util_report = x[j]*v1_i + y[j]*v2_i - p[j]
                    util_truth = x[i]*v1_i + y[i]*v2_i - p[i]
                    if util_report - util_truth > 1e-8:
                        ic_violated = True
                        break
                if ic_violated:
                    break
            
            if ic_violated:
                # Fallback to trivial solution
                allocations = np.zeros((n_types, 2))
                payments = np.zeros(n_types)
                return {'allocations': allocations, 'payments': payments}
        
        allocations = np.column_stack((x, y))
        return {'allocations': allocations, 'payments': p}
        
    except Exception as e:
        # Fallback: return a trivial feasible solution
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

# EVOLVE-BLOCK-END
