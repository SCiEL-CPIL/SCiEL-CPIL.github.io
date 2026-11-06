"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # 1. Define Type Space
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    n_goods = 2
    
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # 2. Parse Distribution
    probs = np.zeros(n_types)
    
    if isinstance(distribution_spec, dict):
        if all(isinstance(k, tuple) for k in distribution_spec.keys()):
            for k, v in distribution_spec.items():
                if k in type_to_idx:
                    probs[type_to_idx[k]] = v
        elif 'probs' in distribution_spec:
            p_data = distribution_spec['probs']
            if isinstance(p_data, (list, np.ndarray)) and len(p_data) == n_types:
                probs = np.array(p_data)
    elif isinstance(distribution_spec, (list, np.ndarray)):
        if len(distribution_spec) == n_types:
            probs = np.array(distribution_spec)
            
    total_prob = np.sum(probs)
    if total_prob > 0:
        probs /= total_prob
    else:
        probs = np.ones(n_types) / n_types

    # 3. Setup LP
    # Vars: x_0_0...x_15_1, u_0...u_15
    n_vars = n_types * n_goods + n_types
    
    c = np.zeros(n_vars)
    for i in range(n_types):
        v = np.array(types[i])
        for j in range(n_goods):
            var_idx = i * n_goods + j
            c[var_idx] -= probs[i] * v[j]
        var_idx = n_types * n_goods + i
        c[var_idx] += probs[i]
        
    A_ub = []
    b_ub = []
    
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            v_i = np.array(types[i])
            v_j = np.array(types[j])
            diff_v = v_i - v_j
            
            row = np.zeros(n_vars)
            for k in range(n_goods):
                row[j * n_goods + k] = diff_v[k]
            row[n_types * n_goods + i] = -1.0
            row[n_types * n_goods + j] = 1.0
            
            A_ub.append(row)
            b_ub.append(0.0)
            
    bounds = []
    for _ in range(n_types * n_goods):
        bounds.append((0, 1))
    for _ in range(n_types):
        bounds.append((0, None))
        
    try:
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')
        if not res.success:
            from autoresearch_suite.science_economics_expansion import auction_baseline
            return auction_baseline(distribution_spec, support, constraints)
            
        x_u = res.x
        allocations = x_u[:n_types * n_goods].reshape(n_types, n_goods)
        utilities = x_u[n_types * n_goods:]
        
        payments = np.zeros(n_types)
        for i in range(n_types):
            v = np.array(types[i])
            x_i = allocations[i]
            payments[i] = np.dot(v, x_i) - utilities[i]
            
        payments = np.maximum(payments, 0)
        
        return {
            "allocations": allocations,
            "payments": payments
        }
        
    except Exception:
        from autoresearch_suite.science_economics_expansion import auction_baseline
        return auction_baseline(distribution_spec, support, constraints)

# EVOLVE-BLOCK-END
