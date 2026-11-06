# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder, two goods,
    values in {0,1,2,3}, with 16 types.
    
    Returns:
        allocations: (16, 2) array of allocation probabilities
        payments: (16,) array of expected payments
    """
    t_start = time.time()
    deadline = t_start + 80.0
    
    # Parse the 16 types: (v1, v2) where v1, v2 in {0,1,2,3}
    # Type index: idx = v1 * 4 + v2
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    
    # Extract probabilities from distribution_spec
    # distribution_spec could be a dict, list, or array of 16 probabilities
    if isinstance(distribution_spec, dict):
        # Try to extract probabilities
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            # Fallback: uniform
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != n_types:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    probs = np.array(probs, dtype=float).flatten()
    if len(probs) != n_types:
        probs = np.ones(n_types) / n_types
    probs = np.clip(probs, 0, None)
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(n_types) / n_types
    
    # Values for each type
    values = np.array([[v1, v2] for v1, v2 in types], dtype=float)
    
    # Decision variables:
    # x[t, g] = allocation probability of good g to type t, for t in 0..15, g in 0,1
    # p[t] = payment for type t
    #
    # Variables: x[0..15, 0..1] (32 vars), p[0..15] (16 vars) -> 48 vars
    # Let's index: var[i] for i in 0..31 is x[t,g] where t = i//2, g = i%2
    # var[32 + t] is p[t]
    
    n_x = n_types * 2  # 32 allocation vars
    n_p = n_types      # 16 payment vars
    n_vars = n_x + n_p  # 48
    
    # Objective: maximize expected revenue = sum_t probs[t] * p[t]
    # In linprog, we minimize, so c = -[0,0,...,0, -probs[0], -probs[1], ...]
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_x + t] = -probs[t]
    
    # Constraints:
    # 1. Feasibility: 0 <= x[t,g] <= 1 for all t,g
    # 2. Individual rationality: p[t] <= sum_g x[t,g] * v[t,g] for all t
    #    i.e., p[t] - sum_g x[t,g]*v[t,g] <= 0
    # 3. Incentive compatibility (truthfulness): For all t, t' (report t' when type t),
    #    u(t, t') <= u(t, t)
    #    u(t, t') = sum_g x[t',g] * v[t,g] - p[t']
    #    u(t, t) = sum_g x[t,g] * v[t,g] - p[t]
    #    So: sum_g x[t',g] * v[t,g] - p[t'] <= sum_g x[t,g] * v[t,g] - p[t]
    #    Rearranging: sum_g (x[t',g] * v[t,g] - x[t,g] * v[t,g]) - p[t'] + p[t] <= 0
    #    i.e., sum_g v[t,g] * (x[t',g] - x[t,g]) + p[t] - p[t'] <= 0
    
    # Build constraint matrix A_ub and b_ub
    # Constraint 1: x[t,g] <= 1 for all t,g (32 constraints)
    # Constraint 2: p[t] - sum_g x[t,g]*v[t,g] <= 0 for all t (16 constraints)
    # Constraint 3: IC constraints for all t, t' pairs (16*16 = 256 constraints)
    
    n_constraints = n_x + n_p + n_types * n_types  # 32 + 16 + 256 = 304
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    row = 0
    # Constraint 1: x[t,g] <= 1
    for t in range(n_types):
        for g in range(2):
            var_idx = t * 2 + g
            A_ub[row, var_idx] = 1.0
            b_ub[row] = 1.0
            row += 1
    
    # Constraint 2: IR: p[t] - sum_g x[t,g]*v[t,g] <= 0
    for t in range(n_types):
        for g in range(2):
            var_idx = t * 2 + g
            A_ub[row, var_idx] = -values[t, g]
        A_ub[row, n_x + t] = 1.0
        b_ub[row] = 0.0
        row += 1
    
    # Constraint 3: IC: for all t, t'
    for t in range(n_types):
        for t_prime in range(n_types):
            for g in range(2):
                # x[t',g] * v[t,g] - x[t,g] * v[t,g]
                A_ub[row, t_prime * 2 + g] = values[t, g]
                A_ub[row, t * 2 + g] = -values[t, g]
            A_ub[row, n_x + t] = 1.0
            A_ub[row, n_x + t_prime] = -1.0
            b_ub[row] = 0.0
            row += 1
    
    # Bounds: 0 <= x[t,g] <= 1, p[t] >= 0
    bounds = [(0, 1)] * n_x + [(0, None)] * n_p
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.status != 0:
        # Fallback: uniform allocation, zero payments
        allocations = np.ones((n_types, 2)) / 2
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    x_flat = result.x[:n_x].reshape(n_types, 2)
    p = result.x[n_x:]
    
    # Clip to ensure feasibility
    x_flat = np.clip(x_flat, 0, 1)
    p = np.clip(p, 0, None)
    
    return {'allocations': x_flat, 'payments': p}
# EVOLVE-BLOCK-END
