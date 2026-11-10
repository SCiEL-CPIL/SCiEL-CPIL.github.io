"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Generate all 16 types: (v1, v2) where v1, v2 in {0,1,2,3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    type_values = np.array(types, dtype=float)  # shape (16, 2)
    
    # Parse distribution_spec to get probabilities for each type
    # distribution_spec should tell us which distribution to use
    # Based on the task, there are three fixed distributions
    # Let's assume distribution_spec is a string or index indicating which distribution
    
    # Try to parse the distribution
    if isinstance(distribution_spec, str):
        dist_name = distribution_spec
    elif isinstance(distribution_spec, (int, float)):
        dist_name = str(int(distribution_spec))
    else:
        dist_name = str(distribution_spec)
    
    # Define the three fixed distributions
    # We need to figure out what they are. Let's assume they are:
    # "uniform": uniform over all 16 types
    # "independent": independent uniform over each good
    # "correlated": some correlated distribution
    
    # For now, let's try to get probabilities from distribution_spec
    # If it's a dict, use it directly
    if isinstance(distribution_spec, dict):
        # Assume keys are type indices or (v1,v2) tuples
        probs = np.zeros(16)
        for key, val in distribution_spec.items():
            if isinstance(key, tuple):
                v1, v2 = key
                idx = v1 * 4 + v2
                probs[idx] = val
            elif isinstance(key, int):
                probs[key] = val
        # Normalize
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(16) / 16
    else:
        # Default to uniform if we can't parse
        probs = np.ones(16) / 16
    
    # LP formulation
    # Variables: x1[0..15], x2[0..15], p[0..15] -> 48 variables
    # Objective: maximize sum_i probs[i] * p[i]
    # In linprog: minimize -sum_i probs[i] * p[i]
    
    c = np.zeros(48)
    for i in range(16):
        c[32 + i] = -probs[i]  # minimize -revenue = maximize revenue
    
    # Equality constraints: none
    
    # Inequality constraints: A_ub @ x <= b_ub
    # We'll build them up
    
    A_ub_rows = []
    b_ub_vals = []
    
    # x1[i] <= 1 for all i
    for i in range(16):
        row = np.zeros(48)
        row[i] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(1)
    
    # x2[i] <= 1 for all i
    for i in range(16):
        row = np.zeros(48)
        row[16 + i] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(1)
    
    # -x1[i] <= 0 for all i (x1[i] >= 0)
    for i in range(16):
        row = np.zeros(48)
        row[i] = -1
        A_ub_rows.append(row)
        b_ub_vals.append(0)
    
    # -x2[i] <= 0 for all i (x2[i] >= 0)
    for i in range(16):
        row = np.zeros(48)
        row[16 + i] = -1
        A_ub_rows.append(row)
        b_ub_vals.append(0)
    
    # -p[i] <= 0 for all i (p[i] >= 0)
    for i in range(16):
        row = np.zeros(48)
        row[32 + i] = -1
        A_ub_rows.append(row)
        b_ub_vals.append(0)
    
    # IR: v1[i]*x1[i] + v2[i]*x2[i] - p[i] >= 0
    # => -v1[i]*x1[i] - v2[i]*x2[i] + p[i] <= 0
    for i in range(16):
        v1, v2 = type_values[i]
        row = np.zeros(48)
        row[i] = -v1
        row[16 + i] = -v2
        row[32 + i] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0)
    
    # IC: v1[i]*x1[i] + v2[i]*x2[i] - p[i] >= v1[i]*x1[j] + v2[i]*x2[j] - p[j]
    # => v1[i]*x1[i] + v2[i]*x2[i] - p[i] - v1[i]*x1[j] - v2[i]*x2[j] + p[j] >= 0
    # => -v1[i]*x1[i] - v2[i]*x2[i] + p[i] + v1[i]*x1[j] + v2[i]*x2[j] - p[j] <= 0
    for i in range(16):
        v1_i, v2_i = type_values[i]
        for j in range(16):
            if i == j:
                continue
            row = np.zeros(48)
            # -v1_i * x1[i]
            row[i] = -v1_i
            # -v2_i * x2[i]
            row[16 + i] = -v2_i
            # +p[i]
            row[32 + i] = 1
            # +v1_i * x1[j]
            row[j] += v1_i
            # +v2_i * x2[j]
            row[16 + j] += v2_i
            # -p[j]
            row[32 + j] -= 1
            A_ub_rows.append(row)
            b_ub_vals.append(0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs', options={'presolve': True})
    
    if result.success:
        x1 = result.x[0:16]
        x2 = result.x[16:32]
        p = result.x[32:48]
        
        # Clip to ensure feasibility
        x1 = np.clip(x1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        p = np.clip(p, 0, None)
        
        allocations = np.column_stack([x1, x2])
        payments = p
    else:
        # Fallback: simple mechanism
        x1 = np.ones(16)
        x2 = np.ones(16)
        p = type_values.sum(axis=1)  # p(v) = v1 + v2
        allocations = np.column_stack([x1, x2])
        payments = p
    
    # Check deadline
    if time.time() > deadline:
        pass  # Return what we have
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
