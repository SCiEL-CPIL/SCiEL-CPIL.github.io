"""Direct convex LP over allocation and payment variables for optimal finite-support mechanism."""
import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: support of the value space
        constraints: additional constraints
    
    Returns:
        dict with 'allocations': (16, 2) array and 'payments': (16,) array
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Parse distribution
    values = [0, 1, 2, 3]
    
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
    
    # Parse support
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]
    
    n = 16
    assert len(types) == n, f"Expected 16 types, got {len(types)}"
    
    # Value matrix: val[t] = (v1, v2)
    val_matrix = np.array(types, dtype=float)  # shape (16, 2)

    # Variables:
    # x[t, g] for t in 0..15, g in 0..1 -> 32 variables
    # p[t] for t in 0..15 -> 16 variables
    # Total variables = 48
    # Order: x[0,0], x[0,1], x[1,0], x[1,1], ..., p[0], p[1], ...
    
    n_vars = n * 2 + n
    
    # Objective: Maximize sum_t probs[t] * p[t]
    # Minimize -sum_t probs[t] * p[t]
    c = np.zeros(n_vars)
    for t in range(n):
        c[n*2 + t] = -probs[t]
    
    # Constraints:
    # 1. IC: p[t] - p[t'] <= v[t]·x[t] - v[t]·x[t']
    #    p[t] - p[t'] - (v[t,0]*x[t,0] + v[t,1]*x[t,1]) + (v[t,0]*x[t',0] + v[t,1]*x[t',1]) <= 0
    #    Wait, the standard IC is:
    #    v[t]·x[t] - p[t] >= v[t]·x[t'] - p[t']
    #    p[t] - p[t'] <= v[t]·x[t] - v[t]·x[t']
    #    p[t] - p[t'] - v[t]·x[t] + v[t]·x[t'] <= 0
    
    A_rows = []
    b_rows = []
    
    for t in range(n):
        for t2 in range(n):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            # p[t] coeff is 1
            row[n*2 + t] = 1.0
            # p[t2] coeff is -1
            row[n*2 + t2] = -1.0
            
            # x[t, g] coeff is -v[t, g]
            # x[t2, g] coeff is +v[t, g]
            for g in range(2):
                row[t*2 + g] = -val_matrix[t, g]
                row[t2*2 + g] = val_matrix[t, g]
            
            A_rows.append(row)
            b_rows.append(0.0)
            
    # 2. IR: p[t] <= v[t]·x[t]
    #    p[t] - v[t]·x[t] <= 0
    for t in range(n):
        row = np.zeros(n_vars)
        row[n*2 + t] = 1.0
        for g in range(2):
            row[t*2 + g] = -val_matrix[t, g]
        A_rows.append(row)
        b_rows.append(0.0)
        
    A_ub = np.array(A_rows) if A_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_rows) if b_rows else np.zeros(0)
    
    # Bounds:
    # x[t, g] in [0, 1]
    # p[t] in [0, None] (Non-negative payments)
    bounds = []
    for t in range(n):
        for g in range(2):
            bounds.append((0, 1))
    for t in range(n):
        bounds.append((0, None))
        
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            # Extract allocations and payments
            alloc = result.x[:n*2].reshape(n, 2)
            pay = result.x[n*2:n*2+n]
            
            # Clean up numerical noise
            alloc = np.clip(alloc, 0, 1)
            pay = np.maximum(pay, 0)
            
            return {'allocations': alloc, 'payments': pay}
        else:
            # Fallback if LP fails (should not happen for feasible problems)
            return {'allocations': np.zeros((n, 2)), 'payments': np.zeros(n)}
    except Exception:
        return {'allocations': np.zeros((n, 2)), 'payments': np.zeros(n)}
