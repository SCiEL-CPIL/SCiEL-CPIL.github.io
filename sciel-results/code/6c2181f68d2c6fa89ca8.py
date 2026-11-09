"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct_auction(distribution_spec, support, constraints):
    import numpy as np
    from scipy.optimize import linprog
    
    # Values are in {0, 1, 2, 3} for each of 2 goods
    values = [0, 1, 2, 3]
    n_vals = len(values)
    n_types = n_vals * n_vals  # 16 types
    
    # Enumerate all types: (v1, v2)
    types = [(v1, v2) for v1 in values for v2 in values]
    type_idx = {t: i for i, t in enumerate(types)}
    
    # Parse distribution_spec to get probabilities for each type
    # distribution_spec could be various formats; assume it's a dict or array
    if isinstance(distribution_spec, dict):
        # Could be {type: prob} or nested
        if all(k in type_idx for k in distribution_spec.keys()):
            probs = np.array([distribution_spec.get(t, 0.0) for t in types])
        else:
            # Maybe nested dict or list
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == n_types:
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    else:
        probs = np.ones(n_types) / n_types
    
    # Normalize probabilities
    prob_sum = probs.sum()
    if prob_sum > 0:
        probs = probs / prob_sum
    else:
        probs = np.ones(n_types) / n_types
    
    # Variables: q1[t], q2[t], p[t] for each type t
    # Total variables: 3 * 16 = 48
    # x[0:16] = q1, x[16:32] = q2, x[32:48] = p
    
    n_vars = 3 * n_types
    
    # Objective: maximize expected revenue = sum_t probs[t] * p[t]
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[32 + t] = -probs[t]  # maximize p[t] * probs[t]
    
    # Constraints:
    # 1. 0 <= q1[t] <= 1, 0 <= q2[t] <= 1
    # 2. Monotonicity: q1[v1, v2] <= q1[v1+1, v2] for v1 in {0,1,2}
    #    q2[v1, v2] <= q2[v1, v2+1] for v2 in {0,1,2}
    # 3. IR: p[t] <= v1[t]*q1[t] + v2[t]*q2[t] for all t
    # 4. Nonnegative payments: p[t] >= 0
    
    A_ub = []
    b_ub = []
    
    # Monotonicity for q1: q1[v1, v2] - q1[v1+1, v2] <= 0
    for v2 in values:
        for v1 in range(n_vals - 1):
            t1 = type_idx[(v1, v2)]
            t2 = type_idx[(v1 + 1, v2)]
            row = np.zeros(n_vars)
            row[0 + t1] = 1   # q1[t1]
            row[0 + t2] = -1  # -q1[t2]
            A_ub.append(row)
            b_ub.append(0)
    
    # Monotonicity for q2: q2[v1, v2] - q2[v1, v2+1] <= 0
    for v1 in values:
        for v2 in range(n_vals - 1):
            t1 = type_idx[(v1, v2)]
            t2 = type_idx[(v1, v2 + 1)]
            row = np.zeros(n_vars)
            row[16 + t1] = 1   # q2[t1]
            row[16 + t2] = -1  # -q2[t2]
            A_ub.append(row)
            b_ub.append(0)
    
    # IR: p[t] - v1[t]*q1[t] - v2[t]*q2[t] <= 0
    for t in range(n_types):
        v1, v2 = types[t]
        row = np.zeros(n_vars)
        row[32 + t] = 1        # p[t]
        row[0 + t] = -v1       # -v1*q1[t]
        row[16 + t] = -v2      # -v2*q2[t]
        A_ub.append(row)
        b_ub.append(0)
    
    # Nonnegative payments: -p[t] <= 0
    for t in range(n_types):
        row = np.zeros(n_vars)
        row[32 + t] = -1
        A_ub.append(row)
        b_ub.append(0)
    
    A_ub = np.array(A_ub) if A_ub else np.array([]).reshape(0, n_vars)
    b_ub = np.array(b_ub) if b_ub else np.array([])
    
    # Bounds: 0 <= q1, q2 <= 1, 0 <= p
    bounds = []
    for t in range(n_types):
        bounds.append((0, 1))   # q1
    for t in range(n_types):
        bounds.append((0, 1))   # q2
    for t in range(n_types):
        bounds.append((0, None))  # p
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.status != 0:
        # Fallback: allocate both goods to everyone, charge 0
        allocations = np.ones((n_types, 2))
        payments = np.zeros(n_types)
    else:
        x = result.x
        q1 = x[0:n_types]
        q2 = x[n_types:2*n_types]
        p = x[2*n_types:3*n_types]
        
        allocations = np.column_stack([q1, q2])
        payments = p
    
    return {"allocations": allocations, "payments": payments}
# EVOLVE-BLOCK-END

