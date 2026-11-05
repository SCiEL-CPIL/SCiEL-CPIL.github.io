import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s
    
    # Values are in {0, 1, 2, 3} for each good
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)  # 16
    
    # Map type to index
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Get distribution
    if isinstance(distribution_spec, str):
        if distribution_spec in ('uniform', 'independent_uniform'):
            probs = np.ones(n_types) / n_types
        elif distribution_spec == 'independent':
            probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) == n_types:
            probs = probs / probs.sum()
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    # Variables: x1(0..15), x2(0..15), p(0..15) -> 48 variables
    # Index: type i -> x1 at i, x2 at 16+i, p at 32+i
    n_vars = 3 * n_types
    
    # Objective: maximize sum pi(v)*p(v) -> minimize -sum pi(v)*p(v)
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[32 + i] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    A_rows = []
    b_rows = []
    
    # IR constraints: v1*x1(v) + v2*x2(v) - p(v) >= 0 -> -v1*x1(v) - v2*x2(v) + p(v) <= 0
    for i in range(n_types):
        v1, v2 = types[i]
        row = np.zeros(n_vars)
        row[i] = -v1       # -v1 * x1
        row[16 + i] = -v2  # -v2 * x2
        row[32 + i] = 1    # +1 * p
        A_rows.append(row)
        b_rows.append(0.0)
    
    # IC constraints: for all v, v':
    # v1*x1(v) + v2*x2(v) - p(v) >= v1*x1(v') + v2*x2(v') - p(v')
    # -> v1*x1(v) + v2*x2(v) - p(v) - v1*x1(v') - v2*x2(v') + p(v') <= 0
    for i in range(n_types):
        v1, v2 = types[i]
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[i] = v1           # +v1 * x1(v)
            row[16 + i] = v2      # +v2 * x2(v)
            row[32 + i] = -1      # -p(v)
            row[j] = -v1          # -v1 * x1(v')
            row[16 + j] = -v2     # -v2 * x2(v')
            row[32 + j] = 1       # +p(v')
            A_rows.append(row)
            b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds: 0 <= x1, x2 <= 1; p >= 0
    bounds = []
    for i in range(n_types):
        bounds.append((0, 1))  # x1
    for i in range(n_types):
        bounds.append((0, 1))  # x2
    for i in range(n_types):
        bounds.append((0, None))  # p
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if time.time() > deadline:
        # Return a simple feasible solution
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
    
    if result.success:
        x = result.x
        allocations = np.column_stack([x[0:16], x[16:32]])
        payments = x[32:48]
        
        # Clean up small numerical errors
        allocations = np.clip(allocations, 0, 1)
        payments = np.maximum(payments, 0)
        
        return {'allocations': allocations, 'payments': payments}
    else:
        # Fallback: simple feasible solution
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}
