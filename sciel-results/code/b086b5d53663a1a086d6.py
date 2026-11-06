"""Incentive-compatible finite auction design via direct LP solve."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction that maximizes expected revenue.
    
    Args:
        distribution_spec: Dict with 'probs' (list of 16 floats) or similar structure.
        support: List of (v1, v2) tuples, likely [(0,0), ..., (3,3)].
        constraints: Dict with constraints (may be empty).
    
    Returns:
        dict: { 'allocations': np.array of shape (16, 2), 
                'payments': np.array of shape (16,) }
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Parse the support and distribution
    # support should be 16 (v1, v2) pairs where v1, v2 in {0,1,2,3}
    if support is None:
        support = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    
    n = len(support)
    
    # Get probabilities
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'prob' in distribution_spec:
            probs = np.array(distribution_spec['prob'], dtype=float)
        else:
            probs = np.ones(n) / n
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n) / n
    
    # Normalize probabilities
    probs = probs / probs.sum()
    
    # Values for each type
    values = np.array(support, dtype=float)  # shape (n, 2)
    
    # For each type i, the value of good j is values[i, j]
    
    # LP formulation:
    # Variables: x[i, j] for i in 0..n-1, j in 0,1 (allocation probabilities)
    #            p[i] for i in 0..n-1 (payments)
    # Total variables: 2*n + n = 3*n
    
    # Objective: maximize sum_i probs[i] * sum_j x[i,j] * values[i,j] - sum_i probs[i] * p[i]
    # Equivalently: minimize -sum_i probs[i] * sum_j x[i,j] * values[i,j] + sum_i probs[i] * p[i]
    
    # Constraints:
    # 1. Feasibility: 0 <= x[i,j] <= 1 for all i,j
    #    x[i,j] >= 0 (lower bound)
    #    x[i,j] <= 1 (upper bound)
    #    p[i] >= 0 (lower bound)
    
    # 2. Individual Rationality (IR):
    #    For each type i: sum_j x[i,j]*values[i,j] - p[i] >= 0
    #    i.e., p[i] <= sum_j x[i,j]*values[i,j]
    #    -sum_j x[i,j]*values[i,j] + p[i] <= 0
    
    # 3. Incentive Compatibility (IC) / Truthfulness:
    #    For each pair of types (i, i'), type i should prefer reporting i over i':
    #    sum_j x[i,j]*values[i,j] - p[i] >= sum_j x[i',j]*values[i,j] - p[i']
    #    i.e., p[i'] - p[i] >= sum_j (x[i',j] - x[i,j])*values[i,j]
    #    Rearranging: -sum_j (x[i',j] - x[i,j])*values[i,j] + p[i'] - p[i] <= 0
    #    Or: sum_j (x[i,j] - x[i',j])*values[i,j] - p[i] + p[i'] <= 0
    
    # Let's set up the LP with scipy.optimize.linprog
    # Variables: [x[0,0], x[0,1], x[1,0], x[1,1], ..., x[n-1,0], x[n-1,1], p[0], p[1], ..., p[n-1]]
    # Index: x[i,j] is at index 2*i + j, p[i] is at index 2*n + i
    
    # Objective coefficients (minimize)
    c = np.zeros(3 * n)
    for i in range(n):
        for j in range(2):
            c[2*i + j] = -probs[i] * values[i, j]
        c[2*n + i] = probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    # We'll build A_ub and b_ub row by row
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IR constraints: for each i
    # -sum_j x[i,j]*values[i,j] + p[i] <= 0
    for i in range(n):
        row = np.zeros(3 * n)
        for j in range(2):
            row[2*i + j] = -values[i, j]
        row[2*n + i] = 1.0
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IC constraints: for each pair (i, i')
    # sum_j (x[i,j] - x[i',j])*values[i,j] - p[i] + p[i'] <= 0
    for i in range(n):
        for ip in range(n):
            if i == ip:
                continue
            row = np.zeros(3 * n)
            for j in range(2):
                row[2*i + j] = values[i, j]
                row[2*ip + j] = -values[i, j]
            row[2*n + i] = -1.0
            row[2*n + ip] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds:
    # x[i,j] in [0, 1]
    # p[i] in [0, inf]
    bounds = []
    for i in range(n):
        for j in range(2):
            bounds.append((0.0, 1.0))
    for i in range(n):
        bounds.append((0.0, None))
    
    # Solve the LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: return zero allocations and zero payments
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {'allocations': allocations, 'payments': payments}
    
    x = result.x
    allocations = np.zeros((n, 2))
    for i in range(n):
        for j in range(2):
            allocations[i, j] = x[2*i + j]
    payments = np.zeros(n)
    for i in range(n):
        payments[i] = x[2*n + i]
    
    # Clean up small numerical errors
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
