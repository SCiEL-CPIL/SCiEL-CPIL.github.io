# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 80.0
    
    # 16 types: (v1, v2) with v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    
    # Extract probabilities from distribution_spec
    probs = _get_probs(distribution_spec, n_types)
    
    # LP variables: x1[0..15], x2[0..15], u[0..15]
    # Total: 48 variables
    # Objective: max sum_t probs[t] * (v1[t]*x1[t] + v2[t]*x2[t] - u[t])
    # = max sum_t probs[t]*v1[t]*x1[t] + probs[t]*v2[t]*x2[t] - probs[t]*u[t]
    
    c = np.zeros(3 * n_types)
    for t in range(n_types):
        v1, v2 = types[t]
        c[t] = probs[t] * v1           # x1[t]
        c[n_types + t] = probs[t] * v2 # x2[t]
        c[2 * n_types + t] = -probs[t] # u[t]
    
    # Inequality constraints: A_ub @ vars <= b_ub
    # 1) IC constraints: u[t] - u[t'] >= v[t]·x[t'] - v[t']·x[t']
    #    => -u[t] + u[t'] + v[t]·x[t'] - v[t']·x[t'] <= 0
    #    For each pair (t, t'), t != t'
    
    # 2) IR: u[t] >= 0 => -u[t] <= 0
    # 3) Feasibility: 0 <= x1[t] <= 1, 0 <= x2[t] <= 1
    #    x1[t] <= 1 => x1[t] <= 1
    #    -x1[t] <= 0
    #    Similarly for x2
    
    # Build A_ub
    rows = []
    
    # IC constraints: for all t, t' with t != t'
    for t in range(n_types):
        for tp in range(n_types):
            if t == tp:
                continue
            v1_t, v2_t = types[t]
            v1_tp, v2_tp = types[tp]
            row = np.zeros(3 * n_types)
            # -u[t] + u[tp] + v1_t*x1[tp] + v2_t*x2[tp] - v1_tp*x1[tp] - v2_tp*x2[tp] <= 0
            row[2 * n_types + t] = -1.0
            row[2 * n_types + tp] = 1.0
            row[tp] = v1_t - v1_tp
            row[n_types + tp] = v2_t - v2_tp
            rows.append(row)
    
    # IR constraints: -u[t] <= 0
    for t in range(n_types):
        row = np.zeros(3 * n_types)
        row[2 * n_types + t] = -1.0
        rows.append(row)
    
    # x1[t] <= 1
    for t in range(n_types):
        row = np.zeros(3 * n_types)
        row[t] = 1.0
        rows.append(row)
    
    # -x1[t] <= 0
    for t in range(n_types):
        row = np.zeros(3 * n_types)
        row[t] = -1.0
        rows.append(row)
    
    # x2[t] <= 1
    for t in range(n_types):
        row = np.zeros(3 * n_types)
        row[n_types + t] = 1.0
        rows.append(row)
    
    # -x2[t] <= 0
    for t in range(n_types):
        row = np.zeros(3 * n_types)
        row[n_types + t] = -1.0
        rows.append(row)
    
    A_ub = np.array(rows)
    b_ub = np.zeros(len(rows))
    
    # Bounds: all variables >= 0 (u >= 0, x >= 0), no upper bound in bounds (handled by A_ub)
    bounds = [(0, None)] * (3 * n_types)
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: return all zeros (valid but zero revenue)
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    allocations = np.array([[x[t], x[n_types + t]] for t in range(n_types)])
    payments = np.array([types[t][0] * x[t] + types[t][1] * x[n_types + t] - x[2 * n_types + t] for t in range(n_types)])
    
    # Clip small numerical errors
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    return {'allocations': allocations, 'payments': payments}


def _get_probs(distribution_spec, n_types):
    """Extract probability vector from distribution_spec."""
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            p = np.array(distribution_spec['probs'], dtype=float)
            if p.shape == (n_types,):
                return p / p.sum()
            if p.shape == (4, 4):
                return p.flatten() / p.sum()
        if 'prob' in distribution_spec:
            p = np.array(distribution_spec['prob'], dtype=float)
            if p.shape == (n_types,):
                return p / p.sum()
            if p.shape == (4, 4):
                return p.flatten() / p.sum()
        if 'pmf' in distribution_spec:
            p = np.array(distribution_spec['pmf'], dtype=float)
            if p.shape == (n_types,):
                return p / p.sum()
            if p.shape == (4, 4):
                return p.flatten() / p.sum()
        # Try to find any array-like value
        for key, val in distribution_spec.items():
            if isinstance(val, (list, tuple, np.ndarray)):
                arr = np.array(val, dtype=float)
                if arr.size == n_types:
                    return arr.flatten() / arr.sum()
                if arr.size == 16:
                    return arr.flatten() / arr.sum()
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.size == n_types:
            return arr.flatten() / arr.sum()
        if arr.size == 16:
            return arr.flatten() / arr.sum()
    
    # Default: uniform
    return np.ones(n_types) / n_types

# EVOLVE-BLOCK-END
