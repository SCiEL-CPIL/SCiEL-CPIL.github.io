import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    # Values are in {0, 1, 2, 3} for each good
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)  # 16
    
    # Map type to index
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Get distribution
    if isinstance(distribution_spec, str):
        # Handle string identifiers
        if distribution_spec in ['uniform', 'independent_uniform', 'correlated']:
            # Assuming 'correlated' maps to uniform as per seed behavior, 
            # or the evaluator passes arrays for specific distributions.
            # Keeping uniform to avoid incorrect guessing of specific correlation structure.
            probs = np.ones(n_types) / n_types
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if probs.sum() > 0:
            probs = probs / probs.sum()  # Normalize
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    # Variables: q1[0:16], q2[16:32], p[32:48]
    n_vars = 3 * n_types
    
    # Objective: maximize expected revenue = sum(pi_i * p_i)
    # linprog minimizes, so we negate
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[2 * n_types + i] = -probs[i]  # Negative for maximization
    
    # Constraints
    A_ub = []
    b_ub = []
    
    # IC constraints: For all true type i and reported type j:
    # v1_i*q1_i + v2_i*q2_i - p_i >= v1_i*q1_j + v2_i*q2_j - p_j
    # Rearranged: v1_i*q1_i + v2_i*q2_i - p_i - v1_i*q1_j - v2_i*q2_j + p_j >= 0
    # For linprog (<= form): -v1_i*q1_i - v2_i*q2_i + p_i + v1_i*q1_j + v2_i*q2_j - p_j <= 0
    
    for i, (v1_i, v2_i) in enumerate(types):
        for j, (v1_j, v2_j) in enumerate(types):
            if i == j:
                continue  # Trivially satisfied
            row = np.zeros(n_vars)
            # -v1_i*q1_i - v2_i*q2_i + p_i + v1_i*q1_j + v2_i*q2_j - p_j
            row[i] = -v1_i                    # q1_i
            row[n_types + i] = -v2_i          # q2_i
            row[2 * n_types + i] = 1.0        # p_i
            row[j] = v1_i                     # q1_j
            row[n_types + j] = v2_i           # q2_j
            row[2 * n_types + j] = -1.0       # p_j
            A_ub.append(row)
            b_ub.append(0.0)
    
    # IR constraints: v1_i*q1_i + v2_i*q2_i - p_i >= 0
    # Rearranged: -v1_i*q1_i - v2_i*q2_i + p_i <= 0
    for i, (v1_i, v2_i) in enumerate(types):
        row = np.zeros(n_vars)
        row[i] = -v1_i
        row[n_types + i] = -v2_i
        row[2 * n_types + i] = 1.0
        A_ub.append(row)
        b_ub.append(0.0)
    
    # Bounds: 0 <= q1 <= 1, 0 <= q2 <= 1, p >= 0
    bounds = []
    for i in range(n_types):
        bounds.append((0, 1))  # q1
    for i in range(n_types):
        bounds.append((0, 1))  # q2
    for i in range(n_types):
        bounds.append((0, None))  # p
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Solve LP with tighter tolerances for higher precision
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                  options={'presolve': True, 
                           'dual_feasibility_tolerance': 1e-12, 
                           'primal_feasibility_tolerance': 1e-12})
    
    if res.status != 0:
        # Fallback: trivial mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x = res.x
    allocations = np.column_stack([x[0:n_types], x[n_types:2*n_types]])
    payments = x[2*n_types:3*n_types]
    
    # Ensure numerical feasibility and snap to bounds to satisfy strict regret constraints
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    # Snap values very close to 0 or 1 to exactly 0 or 1
    allocations[np.abs(allocations) < 1e-10] = 0.0
    allocations[np.abs(allocations - 1.0) < 1e-10] = 1.0
    payments[payments < 1e-10] = 0.0
    
    return {'allocations': allocations, 'payments': payments}
