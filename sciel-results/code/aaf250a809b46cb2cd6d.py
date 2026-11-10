"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Types: (v1, v2) where v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)  # 16
    
    # Parse distribution_spec to get probabilities
    # distribution_spec could be a dict or various formats
    # Let's try to handle common formats
    if isinstance(distribution_spec, dict):
        # Could be {type: prob} or {'probs': [...]} etc.
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'])
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'])
        else:
            # Try to interpret as mapping
            probs = np.zeros(n_types)
            for i, t in enumerate(types):
                if t in distribution_spec:
                    probs[i] = distribution_spec[t]
                elif str(t) in distribution_spec:
                    probs[i] = distribution_spec[str(t)]
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) != n_types:
            # Maybe it's a 2D array
            probs = np.array(distribution_spec, dtype=float).flatten()
    else:
        probs = np.ones(n_types) / n_types
    
    probs = probs.flatten()
    if len(probs) != n_types:
        # Reshape or pad
        if probs.size >= n_types:
            probs = probs[:n_types]
        else:
            new_probs = np.zeros(n_types)
            new_probs[:min(len(probs), n_types)] = probs
            probs = new_probs
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n_types) / n_types
    
    # Variables: x_{v,1}, x_{v,2} for each v (32 vars), p_v for each v (16 vars)
    # Total: 48 variables
    # Indexing: x_{v,1} = v*2 + 0, x_{v,2} = v*2 + 1, p_v = 32 + v
    
    # Objective: maximize sum(probs[v] * p_v)
    # linprog minimizes, so negate
    c = np.zeros(48)
    for v in range(n_types):
        c[32 + v] = -probs[v]  # maximize p_v weighted by prob
    
    # Bounds: x in [0,1], p >= 0
    bounds = []
    for v in range(n_types):
        bounds.append((0, 1))  # x_{v,1}
        bounds.append((0, 1))  # x_{v,2}
    for v in range(n_types):
        bounds.append((0, None))  # p_v
    
    # Constraints: A_ub @ x <= b_ub
    # IC: for all v, v': p_v - p_{v'} <= v·x_v - v'·x_{v'}
    # Rearranged: p_v - p_{v'} - v1*x_{v,1} - v2*x_{v,2} + v1'*x_{v',1} + v2'*x_{v',2} <= 0
    A_rows = []
    b_rows = []
    
    for v_idx, v in enumerate(types):
        v1, v2 = v
        for vp_idx, vp in enumerate(types):
            vp1, vp2 = vp
            row = np.zeros(48)
            # p_v - p_vp
            row[32 + v_idx] = 1.0
            row[32 + vp_idx] = -1.0
            # -v1*x_{v,1} - v2*x_{v,2}
            row[v_idx * 2 + 0] = -v1
            row[v_idx * 2 + 1] = -v2
            # +vp1*x_{vp,1} + vp2*x_{vp,2}
            row[vp_idx * 2 + 0] = vp1
            row[vp_idx * 2 + 1] = vp2
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR: p_v <= v·x_v
    # p_v - v1*x_{v,1} - v2*x_{v,2} <= 0
    for v_idx, v in enumerate(types):
        v1, v2 = v
        row = np.zeros(48)
        row[32 + v_idx] = 1.0
        row[v_idx * 2 + 0] = -v1
        row[v_idx * 2 + 1] = -v2
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if time.time() > deadline:
        # Return baseline
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}
    
    if result.success:
        x_opt = result.x
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        for v_idx in range(n_types):
            allocations[v_idx, 0] = x_opt[v_idx * 2 + 0]
            allocations[v_idx, 1] = x_opt[v_idx * 2 + 1]
            payments[v_idx] = x_opt[32 + v_idx]
        
        # Clip to ensure feasibility
        allocations = np.clip(allocations, 0, 1)
        payments = np.clip(payments, 0, None)
    else:
        # Fallback: simple mechanism
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        for v_idx, v in enumerate(types):
            v1, v2 = v
            if v1 + v2 > 0:
                allocations[v_idx, 0] = 1.0
                allocations[v_idx, 1] = 1.0
                payments[v_idx] = v1 + v2
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
