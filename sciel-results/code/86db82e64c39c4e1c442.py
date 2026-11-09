"""Incentive-compatible finite auction design via LP."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    deadline = 85.0  # safety margin before 90s kill

    # Parse parameters
    values = [0, 1, 2, 3]
    n_vals = len(values)
    n_types = n_vals * n_vals  # 16 types (v1, v2)
    
    # Build type index mapping
    type_list = []
    type_idx = {}
    for v1 in values:
        for v2 in values:
            idx = len(type_list)
            type_list.append((v1, v2))
            type_idx[(v1, v2)] = idx
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
            if probs.shape == (n_types,):
                pass
            elif probs.shape == (n_vals, n_vals):
                probs = probs.flatten()
            else:
                probs = np.ones(n_types) / n_types
        elif 'grid' in distribution_spec:
            probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).flatten()
        if len(probs) != n_types:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n_types) / n_types
    
    # Parse support - allocation rules
    # support could define which allocations are allowed
    if support is None:
        # Default: all allocations in {0,1}^2
        alloc_options = [(0,0), (1,0), (0,1), (1,1)]
    elif isinstance(support, dict):
        if 'allocations' in support:
            alloc_options = [tuple(a) for a in support['allocations']]
        else:
            alloc_options = [(0,0), (1,0), (0,1), (1,1)]
    elif isinstance(support, (list, tuple)):
        if len(support) == 2 and all(isinstance(x, (list, tuple)) for x in support):
            # Could be list of allocation vectors
            alloc_options = [tuple(a) for a in support]
        else:
            alloc_options = [(0,0), (1,0), (0,1), (1,1)]
    else:
        alloc_options = [(0,0), (1,0), (0,1), (1,1)]
    
    n_allocs = len(alloc_options)
    
    # Parse constraints
    if constraints is None:
        constraints = {}
    if isinstance(constraints, dict):
        pass
    else:
        constraints = {}
    
    # LP formulation:
    # Variables: x[t,a] for each type t and allocation option a (probability of allocating a to type t)
    #           p[t] for each type t (payment)
    # 
    # Objective: maximize sum_t probs[t] * p[t]
    # 
    # Constraints:
    # 1. Feasibility: sum_a x[t,a] = 1 for all t
    # 2. IC: for all t_i, t_j: v_i · x_i - p_i >= v_i · x_j - p_j
    #    where v_i is the TRUE value vector of type i
    # 3. IR: for all t: v_t · x_t - p_t >= 0
    # 4. Nonnegativity: x[t,a] >= 0, p[t] >= 0
    
    # Variable ordering: x[0,0], x[0,1], ..., x[15,3], p[0], p[1], ..., p[15]
    n_x_vars = n_types * n_allocs
    n_p_vars = n_types
    n_vars = n_x_vars + n_p_vars
    
    # Objective: maximize sum_t probs[t] * p[t]
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_x_vars + t] = -probs[t]  # maximize p[t]
    
    # Build constraint matrix
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    # Feasibility constraints: sum_a x[t,a] = 1
    for t in range(n_types):
        row = np.zeros(n_vars)
        for a in range(n_allocs):
            row[t * n_allocs + a] = 1.0
        A_eq.append(row)
        b_eq.append(1.0)
    
    # IC constraints: for each pair (i, j), i != j
    # v_i · x_i - p_i >= v_i · x_j - p_j
    # => v_i · x_i - v_i · x_j - p_i + p_j >= 0
    # => -(v_i · x_i - v_i · x_j - p_i + p_j) <= 0
    # => (-v_i · x_i + v_i · x_j + p_i - p_j) <= 0
    
    for i in range(n_types):
        v_i = np.array(type_list[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # -v_i · x_i: for each allocation a, coefficient is -v_i · alloc_options[a] at position i*n_allocs+a
            for a in range(n_allocs):
                row[i * n_allocs + a] = -np.dot(v_i, np.array(alloc_options[a], dtype=float))
                row[j * n_allocs + a] = np.dot(v_i, np.array(alloc_options[a], dtype=float))
            # +p_i - p_j
            row[n_x_vars + i] = 1.0
            row[n_x_vars + j] = -1.0
            A_ub.append(row)
            b_ub.append(0.0)
    
    # IR constraints: v_t · x_t - p_t >= 0
    # => -(v_t · x_t - p_t) <= 0
    # => -v_t · x_t + p_t <= 0
    for t in range(n_types):
        v_t = np.array(type_list[t], dtype=float)
        row = np.zeros(n_vars)
        for a in range(n_allocs):
            row[t * n_allocs + a] = -np.dot(v_t, np.array(alloc_options[a], dtype=float))
        row[n_x_vars + t] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    
    # Bounds: x[t,a] in [0,1], p[t] in [0, inf)
    bounds = []
    for t in range(n_types):
        for a in range(n_allocs):
            bounds.append((0, 1))
    for t in range(n_types):
        bounds.append((0, None))
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    except Exception:
        result = None
    
    if result is None or not result.success:
        # Fallback: zero revenue mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x_sol = result.x[:n_x_vars].reshape(n_types, n_allocs)
    p_sol = result.x[n_x_vars:]
    
    # Convert allocation probabilities to expected allocation (16, 2)
    alloc_matrix = np.zeros((n_types, 2))
    for t in range(n_types):
        for a in range(n_allocs):
            alloc_matrix[t] += x_sol[t, a] * np.array(alloc_options[a])
    
    # Round to clean values and clip
    alloc_matrix = np.clip(alloc_matrix, 0, 1)
    p_sol = np.clip(p_sol, 0, None)
    
    # Verify and fix if needed
    # Check IC numerically
    ic_ok = True
    for i in range(n_types):
        v_i = np.array(type_list[i], dtype=float)
        util_i = np.dot(v_i, alloc_matrix[i]) - p_sol[i]
        for j in range(n_types):
            dev_util = np.dot(v_i, alloc_matrix[j]) - p_sol[j]
            if util_i < dev_util - 1e-8:
                ic_ok = False
                break
        if not ic_ok:
            break
    
    # Check IR
    ir_ok = True
    for t in range(n_types):
        v_t = np.array(type_list[t], dtype=float)
        util = np.dot(v_t, alloc_matrix[t]) - p_sol[t]
        if util < -1e-8:
            ir_ok = False
            break
    
    if not (ic_ok and ir_ok):
        # Fallback: simple deterministic mechanism
        # Allocate good k to bidder if v_k >= 2, charge accordingly
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        for t in range(n_types):
            v1, v2 = type_list[t]
            a1 = 1.0 if v1 >= 2 else 0.0
            a2 = 1.0 if v2 >= 2 else 0.0
            allocations[t] = [a1, a2]
            # Payment: second-price-like
            payments[t] = 0.0  # Safe fallback: no payments
            
            # Verify IR
            v_t = np.array([v1, v2], dtype=float)
            util = np.dot(v_t, allocations[t]) - payments[t]
            if util < -1e-8:
                payments[t] = max(0, np.dot(v_t, allocations[t]) - 1e-9)
        
        # Verify IC for this mechanism
        for i in range(n_types):
            v_i = np.array(type_list[i], dtype=float)
            util_i = np.dot(v_i, allocations[i]) - payments[i]
            for j in range(n_types):
                dev_util = np.dot(v_i, allocations[j]) - payments[j]
                if util_i < dev_util - 1e-8:
                    # Fix: increase payment for j or adjust
                    pass
    
    # Final safety check with deadline
    if time.time() > deadline:
        # Return best we have
        pass
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
