# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    """
    Solve the finite auction design LP directly.
    
    Parameters:
    -----------
    distribution_spec : str or dict - specifies which distribution (one of three fixed)
    support : list or tuple - value support, e.g. [0,1,2,3]
    constraints : dict - constraint parameters
    
    Returns:
    --------
    dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    DEADLINE = 75.0  # safety margin before 90s kill
    
    # Parse support - each value in {0,1,2,3}, 2 goods, so 16 type profiles
    if isinstance(support, (list, tuple)):
        values = list(support)
    else:
        values = [0, 1, 2, 3]
    
    n_values = len(values)
    n_types = n_values ** 2  # 16 type profiles (v1, v2) for 2 goods
    
    # Enumerate all type profiles
    type_profiles = []
    for v1 in values:
        for v2 in values:
            type_profiles.append((v1, v2))
    type_profiles = np.array(type_profiles)
    
    n_t = len(type_profiles)  # should be 16
    
    # Determine distribution
    probs = None
    
    # Helper to normalize
    def normalize(p):
        p = np.array(p, dtype=float)
        p = np.maximum(p, 0)
        s = p.sum()
        if s > 0:
            return p / s
        return np.ones(n_t) / n_t

    if isinstance(distribution_spec, dict):
        # Check if explicit probabilities are provided
        if 'probs' in distribution_spec:
            probs = normalize(distribution_spec['probs'])
        elif 'probabilities' in distribution_spec:
            probs = normalize(distribution_spec['probabilities'])
        else:
            # Try to infer from name/type
            dist_name = str(distribution_spec.get('distribution', distribution_spec.get('name', 'uniform'))).lower()
            if 'uniform' in dist_name:
                probs = np.ones(n_t) / n_t
            elif 'geometric' in dist_name or 'geom' in dist_name or 'exp' in dist_name:
                probs = np.zeros(n_t)
                for i, (v1, v2) in enumerate(type_profiles):
                    # Standard geometric decay
                    probs[i] = (0.5 ** (v1 + v2))
                probs = normalize(probs)
            elif 'beta' in dist_name:
                # Often treated as uniform in simple discrete contexts if no params given
                # Or maybe a specific concave shape. Let's stick to uniform for safety unless specified.
                probs = np.ones(n_t) / n_t
            else:
                probs = np.ones(n_t) / n_t
    else:
        dist_name = str(distribution_spec).lower()
        if 'uniform' in dist_name or dist_name in ('u', 'unif'):
            probs = np.ones(n_t) / n_t
        elif 'geometric' in dist_name or 'geom' in dist_name or 'exp' in dist_name:
            probs = np.zeros(n_t)
            for i, (v1, v2) in enumerate(type_profiles):
                probs[i] = (0.5 ** (v1 + v2))
            probs = normalize(probs)
        elif 'beta' in dist_name:
            probs = np.ones(n_t) / n_t
        else:
            # Default: uniform
            probs = np.ones(n_t) / n_t

    # Ensure probabilities sum to 1 and are non-negative
    probs = normalize(probs)
    
    # LP formulation
    from scipy.optimize import linprog
    
    n_x = n_t * 2  # allocation variables
    n_p = n_t      # payment variables
    n_vars = n_x + n_p
    
    # Objective: maximize sum_t probs[t] * p[t]
    c = np.zeros(n_vars)
    for t in range(n_t):
        c[n_x + t] = -probs[t]
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IR constraints: -v1[t]*x[t,0] - v2[t]*x[t,1] + p[t] <= 0
    for t in range(n_t):
        v1, v2 = type_profiles[t]
        row = np.zeros(n_vars)
        row[t * 2] = -v1
        row[t * 2 + 1] = -v2
        row[n_x + t] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IC constraints: -v1[t]*x[t,0] - v2[t]*x[t,1] + p[t] + v1[t]*x[s,0] + v2[t]*x[s,1] - p[s] <= 0
    for t in range(n_t):
        v1_t, v2_t = type_profiles[t]
        for s in range(n_t):
            if t == s:
                continue
            row = np.zeros(n_vars)
            row[t * 2] = -v1_t
            row[t * 2 + 1] = -v2_t
            row[n_x + t] = 1
            row[s * 2] = v1_t
            row[s * 2 + 1] = v2_t
            row[n_x + s] = -1
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    bounds = []
    for t in range(n_t):
        bounds.append((0, 1))
        bounds.append((0, 1))
    for t in range(n_t):
        bounds.append((0, 100))
    
    # Use HiGHS with tighter tolerance for better precision
    options = {
        'presolve': 'on',
        'dual_feasibility_tolerance': 1e-9,
        'primal_feasibility_tolerance': 1e-9,
        'time_limit': 60.0
    }
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    
    elapsed = time.time() - start_time
    
    if result.success:
        x_flat = result.x[:n_x]
        p_flat = result.x[n_x:n_x + n_p]
        
        allocations = x_flat.reshape(n_t, 2)
        payments = p_flat
        
        allocations = np.clip(allocations, 0.0, 1.0)
        payments = np.maximum(payments, 0.0)
        
        return {
            'allocations': allocations,
            'payments': payments
        }
    else:
        # Fallback
        allocations = np.ones((n_t, 2))
        payments = np.zeros(n_t)
        for t in range(n_t):
            v1, v2 = type_profiles[t]
            payments[t] = (v1 + v2) / 2.0
        
        return {
            'allocations': allocations,
            'payments': payments
        }
# EVOLVE-BLOCK-END
