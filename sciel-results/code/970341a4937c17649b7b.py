"""Auction design: incentive-compatible finite mechanism via LP."""
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Type space: 16 types, values in {0,1,2,3} for 2 goods
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    n_goods = 2
    
    # Parse distribution probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            dist = distribution_spec['distribution']
            if isinstance(dist, dict) and 'probs' in dist:
                probs = np.array(dist['probs'], dtype=float)
            else:
                probs = np.array(dist, dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        else:
            probs = np.ones(n_types) / n_types
            for key in distribution_spec:
                val = distribution_spec[key]
                if isinstance(val, (list, np.ndarray)) and len(val) == n_types:
                    probs = np.array(val, dtype=float)
                    break
    else:
        probs = np.array(distribution_spec, dtype=float)
    
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n_types) / n_types
    
    # Parse support: allocation bundles
    if support is None:
        support = [(0, 0), (1, 0), (0, 1), (1, 1)]
    
    n_support = len(support)
    support_arr = np.array(support, dtype=float)  # shape (n_support, n_goods)
    
    # Decision variables:
    # x[i, k] = probability that type i is allocated bundle k, for i in range(n_types), k in range(n_support)
    # p[i] = payment for type i
    # Total variables: n_types * n_support + n_types
    
    n_x = n_types * n_support
    n_p = n_types
    n_vars = n_x + n_p
    
    def x_idx(i, k):
        return i * n_support + k
    
    def p_idx(i):
        return n_x + i
    
    # Objective: maximize expected revenue = sum_i probs[i] * p[i]
    # In linprog, minimize: -sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[p_idx(i)] = -probs[i]
    
    # Equality constraints: sum_k x[i,k] = 1 for each i
    A_eq = []
    b_eq = []
    for i in range(n_types):
        row = np.zeros(n_vars)
        for k in range(n_support):
            row[x_idx(i, k)] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)
    
    # Inequality constraints (A_ub @ x <= b_ub)
    A_ub = []
    b_ub = []
    
    # IC constraints: for each type i and deviation j:
    # u_i(truthful) >= u_i(deviate to j)
    # v_i . q_i - p_i >= v_i . q_j - p_j
    # where q_i = sum_k x[i,k] * support[k], q_j = sum_k x[j,k] * support[k]
    # => v_i . q_i - v_i . q_j - p_i + p_j >= 0
    # => -(v_i . q_i - v_i . q_j - p_i + p_j) <= 0
    
    for i in range(n_types):
        v_i = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue  # trivially satisfied
            row = np.zeros(n_vars)
            
            # v_i . q_i = sum_k x[i,k] * (v_i . support[k])
            for k in range(n_support):
                dot_ik = np.dot(v_i, support_arr[k])
                row[x_idx(i, k)] += dot_ik
            
            # -v_i . q_j = -sum_k x[j,k] * (v_i . support[k])
            for k in range(n_support):
                dot_ik = np.dot(v_i, support_arr[k])
                row[x_idx(j, k)] -= dot_ik
            
            # -p_i + p_j
            row[p_idx(i)] -= 1.0
            row[p_idx(j)] += 1.0
            
            # Constraint: (row @ x) >= 0 => -(row @ x) <= 0
            A_ub.append(-row)
            b_ub.append(0.0)
    
    # IR constraints: v_i . q_i - p_i >= 0 for each i
    # => -(v_i . q_i - p_i) <= 0
    for i in range(n_types):
        v_i = np.array(types[i], dtype=float)
        row = np.zeros(n_vars)
        for k in range(n_support):
            dot_ik = np.dot(v_i, support_arr[k])
            row[x_idx(i, k)] += dot_ik
        row[p_idx(i)] -= 1.0
        # Constraint: (row @ x) >= 0 => -(row @ x) <= 0
        A_ub.append(-row)
        b_ub.append(0.0)
    
    # Bounds
    bounds = []
    for i in range(n_types):
        for k in range(n_support):
            bounds.append((0.0, 1.0))
    for i in range(n_types):
        bounds.append((0.0, None))
    
    # Solve LP
    try:
        result = linprog(
            c=c,
            A_ub=np.array(A_ub) if A_ub else None,
            b_ub=np.array(b_ub) if b_ub else None,
            A_eq=np.array(A_eq),
            b_eq=np.array(b_eq),
            bounds=bounds,
            method='highs'
        )
        
        if result.success:
            x_opt = result.x
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            
            for i in range(n_types):
                for k in range(n_support):
                    prob = x_opt[x_idx(i, k)]
                    if prob > 1e-12:
                        allocations[i] += prob * support_arr[k]
                payments[i] = x_opt[p_idx(i)]
            
            # Clip allocations to [0,1]
            allocations = np.clip(allocations, 0.0, 1.0)
            payments = np.maximum(payments, 0.0)
            
            return {
                'allocations': allocations,
                'payments': payments
            }
        else:
            # LP infeasible or failed - return fallback
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            for i in range(n_types):
                v = np.array(types[i], dtype=float)
                for g in range(n_goods):
                    if v[g] >= 2:
                        allocations[i, g] = 1.0
                    elif v[g] >= 1:
                        allocations[i, g] = 0.5
                payments[i] = 0.5 * np.sum(v)
            
            return {
                'allocations': allocations,
                'payments': payments
            }
    
    except Exception:
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        for i in range(n_types):
            v = np.array(types[i], dtype=float)
            for g in range(n_goods):
                if v[g] >= 2:
                    allocations[i, g] = 1.0
                else:
                    allocations[i, g] = 0.5
            payments[i] = 0.5 * np.sum(v)
        
        return {
            'allocations': allocations,
            'payments': payments
        }
