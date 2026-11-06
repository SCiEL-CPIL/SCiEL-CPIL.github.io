# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog
import scipy.sparse as sp

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder, two goods,
    values in {0,1,2,3}, with 16 types.
    
    Returns:
        allocations: (16, 2) array of allocation probabilities
        payments: (16,) array of expected payments
    """
    t_start = time.time()
    deadline = t_start + 80.0
    
    # Parse the 16 types: (v1, v2) where v1, v2 in {0,1,2,3}
    # Type index: idx = v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    n_goods = 2
    
    # Extract probabilities from distribution_spec
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != n_types:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    probs = np.array(probs, dtype=float).flatten()
    if len(probs) != n_types:
        probs = np.ones(n_types) / n_types
    probs = np.clip(probs, 0, None)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(n_types) / n_types
        
    values = np.array(types, dtype=float) # (16, 2)
    
    # Variable mapping:
    # x[t, g] -> index t*2 + g  (0 to 31)
    # u[t]    -> index 32 + (t-1) for t in 1..15 (32 to 46)
    # u[0] is fixed to 0.
    
    n_x_vars = n_types * n_goods
    n_u_vars = n_types - 1
    n_vars = n_x_vars + n_u_vars
    
    # Objective: Maximize expected revenue = E[v_t . x_t - u_t]
    # c vector for linprog (minimize -c)
    c = np.zeros(n_vars)
    for t in range(n_types):
        for g in range(n_goods):
            c[t * n_goods + g] = -probs[t] * values[t, g]
        if t > 0:
            c[n_x_vars + (t - 1)] = probs[t] # Minimize - ( -pi_t u_t ) -> coeff is +pi_t? 
            # Wait. Maximize sum pi_t (v.x - u).
            # Minimize - sum pi_t v.x + sum pi_t u.
            # So coeff for u_t is +pi_t.
            # Correct.
            
    # Constraints:
    # 1. x[t,g] <= 1
    # 2. IC: v_t . x_t - v_t . x_t' - u_t + u_t' <= 0 (with u_0=0)
    
    # Build sparse matrix for A_ub
    rows = []
    cols = []
    data = []
    b_ub_list = []
    
    # 1. x[t,g] <= 1
    for t in range(n_types):
        for g in range(n_goods):
            var_idx = t * n_goods + g
            rows.append(len(b_ub_list))
            cols.append(var_idx)
            data.append(1.0)
            b_ub_list.append(1.0)
            
    # 2. IC constraints
    for t in range(n_types):
        for t_prime in range(n_types):
            if t == t_prime:
                continue
            row_idx = len(b_ub_list)
            
            # Term: v_t . x_t
            for g in range(n_goods):
                if values[t, g] != 0:
                    rows.append(row_idx)
                    cols.append(t * n_goods + g)
                    data.append(values[t, g])
                    
            # Term: - v_t . x_t'
            for g in range(n_goods):
                if values[t, g] != 0:
                    rows.append(row_idx)
                    cols.append(t_prime * n_goods + g)
                    data.append(-values[t, g])
                    
            # Term: - u_t (if t > 0)
            if t > 0:
                u_idx = n_x_vars + (t - 1)
                rows.append(row_idx)
                cols.append(u_idx)
                data.append(-1.0)
                
            # Term: + u_t' (if t_prime > 0)
            if t_prime > 0:
                u_idx = n_x_vars + (t_prime - 1)
                rows.append(row_idx)
                cols.append(u_idx)
                data.append(1.0)
                
            b_ub_list.append(0.0)
            
    A_ub = sp.csr_matrix((data, (rows, cols)), shape=(len(b_ub_list), n_vars))
    b_ub = np.array(b_ub_list)
    
    bounds = [(0, 1)] * n_x_vars + [(0, None)] * n_u_vars
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-9,
                              'primal_feasibility_tolerance': 1e-9})
    
    if result.status != 0:
        allocations = np.ones((n_types, 2)) / 2
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x_flat = result.x[:n_x_vars].reshape(n_types, n_goods)
    u = np.zeros(n_types)
    for t in range(1, n_types):
        u[t] = result.x[n_x_vars + (t - 1)]
        
    # Calculate payments: p_t = v_t . x_t - u_t
    payments = np.zeros(n_types)
    for t in range(n_types):
        payments[t] = np.dot(values[t], x_flat[t]) - u[t]
        
    # Clip with small margin to ensure strict feasibility
    eps = 1e-9
    x_flat = np.clip(x_flat, eps, 1.0 - eps)
    payments = np.clip(payments, eps, None)
    
    return {'allocations': x_flat, 'payments': payments}
# EVOLVE-BLOCK-END
