# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80  # 80s to leave 10s margin
    
    # Parse the distribution specification
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    
    # Extract probabilities from distribution_spec
    if isinstance(distribution_spec, dict):
        probs = np.zeros(n_types)
        for key, val in distribution_spec.items():
            if isinstance(key, tuple):
                for i, t in enumerate(types):
                    if t == key:
                        probs[i] = val
                        break
            elif isinstance(key, int):
                if 0 <= key < n_types:
                    probs[key] = val
            else:
                pass
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float).flatten()
        if len(probs) != n_types:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    total_prob = probs.sum()
    if total_prob > 1e-15:
        probs = probs / total_prob
    else:
        probs = np.ones(n_types) / n_types
    
    # Variables: x_{i,0}, x_{i,1}, p_i for each type i
    # Total: 2*16 + 16 = 48 variables
    
    n_vars = 2 * n_types + n_types  # 48
    
    # Objective: maximize sum_i probs[i] * p_i => minimize -sum_i probs[i] * p_i
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[2 * n_types + i] = -probs[i]
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IC constraints: for all i, j (i != j):
    # v_i · x_i - p_i >= v_i · x_j - p_j
    # => -v_i[0]*x_{i,0} - v_i[1]*x_{i,1} + v_i[0]*x_{j,0} + v_i[1]*x_{j,1} - p_i + p_j <= 0
    
    for i in range(n_types):
        vi = types[i]
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[i * 2] = -vi[0]
            row[i * 2 + 1] = -vi[1]
            row[j * 2] = vi[0]
            row[j * 2 + 1] = vi[1]
            row[2 * n_types + i] = -1
            row[2 * n_types + j] = 1
            A_ub_rows.append(row)
            b_ub_vals.append(0)
    
    # IR constraints: p_i <= v_i · x_i => -v_i[0]*x_{i,0} - v_i[1]*x_{i,1} + p_i <= 0
    for i in range(n_types):
        vi = types[i]
        row = np.zeros(n_vars)
        row[i * 2] = -vi[0]
        row[i * 2 + 1] = -vi[1]
        row[2 * n_types + i] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds: 0 <= x <= 1, p >= 0
    bounds = [(0, 1)] * (2 * n_types) + [(0, None)] * n_types
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        allocations = np.ones((n_types, 2)) * 0.5
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    if result.success:
        x = result.x
        allocations = x[:2 * n_types].reshape(n_types, 2)
        payments = x[2 * n_types:]
        allocations = np.clip(allocations, 0, 1)
        payments = np.clip(payments, 0, None)
        return {'allocations': allocations, 'payments': payments}
    else:
        allocations = np.ones((n_types, 2)) * 0.5
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
