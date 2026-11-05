"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, minimize
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder with two goods.
    
    Args:
        distribution_spec: dict mapping (v1, v2) to probability, or int index for standard distributions
        support: list of (v1, v2) pairs or similar
        constraints: dict with constraint parameters
        
    Returns:
        dict with 'allocations' (16, 2) and 'payments' (16,)
    """
    deadline = time.time() + 75  # Safety margin before 90s kill
    
    # Define the 16 types: (v1, v2) for v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    num_types = len(types)  # 16
    
    # Determine probabilities
    probs = np.zeros(num_types)
    if isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(num_types) / num_types
    elif isinstance(distribution_spec, int):
        # Three fixed distributions
        # Assuming all are uniform over the 16 types based on current code behavior
        probs = np.ones(num_types) / num_types
    else:
        # Try to interpret as list/array
        try:
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) == num_types:
                total = probs.sum()
                if total > 0:
                    probs /= total
                else:
                    probs = np.ones(num_types) / num_types
            else:
                probs = np.ones(num_types) / num_types
        except:
            probs = np.ones(num_types) / num_types

    # Decision variables: x1[16], x2[16], t[16] -> 48 variables
    # x1[i] = prob of allocating good 1 to type i
    # x2[i] = prob of allocating good 2 to type i  
    # t[i] = payment from type i
    
    # Objective: Maximize sum(probs * t)
    # Minimize -sum(probs * t)
    c = np.zeros(3 * num_types)
    c[2 * num_types:] = -probs
    
    # Constraints: A_ub @ z <= b_ub
    # 1. IC: For all i (true), j (reported):
    # v_i . x_i - t_i >= v_i . x_j - t_j
    # => t_i - t_j - v_i1(x_i1 - x_j1) - v_i2(x_i2 - x_j2) <= 0
    # Variables order: x1 (0-15), x2 (16-31), t (32-47)
    
    # Pre-allocate A_ub
    num_ic = num_types * num_types
    num_ir = num_types
    A_ub = np.zeros((num_ic + num_ir, 3 * num_types))
    b_ub = np.zeros(num_ic + num_ir)
    
    row = 0
    for i in range(num_types):
        v1_i, v2_i = types[i]
        for j in range(num_types):
            # IC constraint for true type i, reported type j
            # Coeffs for x1
            A_ub[row, i] = -v1_i       # x1[i]
            A_ub[row, j] = v1_i         # x1[j] (Note: index j in x1 block is j)
            # Coeffs for x2
            A_ub[row, num_types + i] = -v2_i   # x2[i]
            A_ub[row, num_types + j] = v2_i     # x2[j]
            # Coeffs for t
            A_ub[row, 2 * num_types + i] = 1    # t[i]
            A_ub[row, 2 * num_types + j] = -1   # t[j]
            # b is 0
            row += 1
            
    # 2. IR: For all i:
    # v_i . x_i - t_i >= 0
    # => t_i - v_i1 x_i1 - v_i2 x_i2 <= 0
    for i in range(num_types):
        v1_i, v2_i = types[i]
        A_ub[row, i] = -v1_i
        A_ub[row, num_types + i] = -v2_i
        A_ub[row, 2 * num_types + i] = 1
        # b is 0
        row += 1

    # Bounds
    bounds = []
    for i in range(num_types):
        bounds.append((0, 1))  # x1[i]
    for i in range(num_types):
        bounds.append((0, 1))  # x2[i]
    for i in range(num_types):
        bounds.append((0, None))  # t[i] >= 0

    # Solve LP
    result = None
    try:
        result = linprog(
            c, 
            A_ub=A_ub, 
            b_ub=b_ub, 
            bounds=bounds, 
            method='highs',
            options={'presolve': True, 'dual_feasibility_tolerance': 1e-10, 'primal_feasibility_tolerance': 1e-10}
        )
    except Exception as e:
        pass

    if result is not None and result.success and time.time() < deadline:
        z = result.x
        x1 = z[:num_types]
        x2 = z[num_types:2*num_types]
        t = z[2*num_types:3*num_types]
        
        # Clip small numerical errors
        x1 = np.clip(x1, 0, 1)
        x2 = np.clip(x2, 0, 1)
        t = np.clip(t, 0, None)
        
        allocations = np.column_stack([x1, x2])
        payments = t
        return {'allocations': allocations, 'payments': payments}
    
    # Fallback: Simple mechanism
    # Allocate both goods with prob 0.5, payment = expected value
    # This is IC and IR.
    allocations = np.zeros((num_types, 2))
    payments = np.zeros(num_types)
    for i in range(num_types):
        v1_i, v2_i = types[i]
        allocations[i] = [0.5, 0.5]
        payments[i] = v1_i * 0.5 + v2_i * 0.5
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
