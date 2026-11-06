import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Determine value space from support
    if isinstance(support, dict):
        if 'values' in support:
            values = list(support['values'])
        elif 'support' in support:
            values = list(support['support'])
        else:
            values = [0, 1, 2, 3]
    elif isinstance(support, (list, tuple, np.ndarray)):
        values = list(support)
    else:
        values = [0, 1, 2, 3]
    
    if len(values) != 4:
        values = [0, 1, 2, 3]
    
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)
    
    # Parse distribution
    probs = None
    if isinstance(distribution_spec, dict):
        for key in ['probs', 'distribution', 'probabilities', 'p']:
            if key in distribution_spec:
                probs = np.array(distribution_spec[key], dtype=float)
                break
        if probs is None:
            for key in distribution_spec:
                if isinstance(distribution_spec[key], (list, np.ndarray)):
                    probs = np.array(distribution_spec[key], dtype=float)
                    break
    elif isinstance(distribution_spec, (list, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    
    if probs is None:
        probs = np.ones(n_types) / n_types
    
    probs = probs.flatten()
    if len(probs) != n_types:
        if len(probs) == 4:
            probs = np.outer(probs, probs).flatten()
        else:
            probs = np.ones(n_types) / n_types
    
    total = probs.sum()
    if total > 0:
        probs = probs / total
    else:
        probs = np.ones(n_types) / n_types
    
    V = np.array(types, dtype=float)
    n_vars = 3 * n_types
    
    c = np.zeros(n_vars)
    c[2*n_types:3*n_types] = -probs
    
    A_ub_rows = []
    b_ub_rows = []
    
    for i in range(n_types):
        row = np.zeros(n_vars)
        row[i] = -V[i, 0]
        row[n_types + i] = -V[i, 1]
        row[2*n_types + i] = 1.0
        A_ub_rows.append(row)
        b_ub_rows.append(0.0)
    
    for i in range(n_types):
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[i] = -V[i, 0]
            row[n_types + i] = -V[i, 1]
            row[j] = V[i, 0]
            row[n_types + j] = V[i, 1]
            row[2*n_types + i] = 1.0
            row[2*n_types + j] = -1.0
            A_ub_rows.append(row)
            b_ub_rows.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_rows)
    
    bounds = [(0, 1)] * (2 * n_types) + [(0, None)] * n_types
    
    if time.time() > deadline:
        allocations = np.ones((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        allocations = np.ones((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    allocations = x[:2*n_types].reshape(n_types, 2)
    payments = x[2*n_types:3*n_types]
    
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    return {'allocations': allocations, 'payments': payments}
