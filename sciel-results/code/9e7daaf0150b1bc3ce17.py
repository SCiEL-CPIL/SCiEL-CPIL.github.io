"""Incentive-compatible finite auction design via LP."""
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
    DEADLINE = 85.0  # safety margin before 90s kill
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        probs = np.array(distribution_spec.get('probs', [1/16]*16), dtype=float)
    else:
        probs = np.array(distribution_spec, dtype=float)
    
    probs = probs / probs.sum()
    
    # Support: all (v1, v2) for v1, v2 in {0,1,2,3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = 16
    
    # Map type to index
    type_to_idx = {t: i for i, t in enumerate(types)}
    
    # Variables: x1[0..15], x2[0..15] (allocations for each type)
    # Objective: maximize expected revenue
    # E[revenue] = sum_v pi(v) * p(v)
    # p(v) = v1*x1(v) + v2*x2(v) - sum_{t=0}^{v1-1} x1(t, v2) - sum_{t=0}^{v2-1} x2(v1, t)
    
    # Build objective coefficients
    # E[revenue] = sum_v pi(v) * [v1*x1(v) + v2*x2(v) - sum_{t=0}^{v1-1} x1(t,v2) - sum_{t=0}^{v2-1} x2(v1,t)]
    
    c1 = np.zeros(n_types)  # coefficients for x1
    c2 = np.zeros(n_types)  # coefficients for x2
    
    for v_idx, (v1, v2) in enumerate(types):
        pi = probs[v_idx]
        # Contribution to revenue from v1*x1(v): pi * v1
        c1[v_idx] += pi * v1
        c2[v_idx] += pi * v2
        
        # Subtraction: sum_{t=0}^{v1-1} x1(t, v2)
        for t in range(v1):
            t_idx = type_to_idx[(t, v2)]
            c1[t_idx] -= pi
        
        # Subtraction: sum_{t=0}^{v2-1} x2(v1, t)
        for t in range(v2):
            t_idx = type_to_idx[(v1, t)]
            c2[t_idx] -= pi
    
    # linprog minimizes, so negate
    c = np.concatenate([-c1, -c2])
    
    # Constraints:
    # 1. 0 <= x1[v] <= 1, 0 <= x2[v] <= 1 (bounds)
    # 2. IC: for all v, v': u(v, v') <= u(v, v)
    # 3. IR: p(v) >= 0 for all v
    
    # IC constraint: u(v, v') - u(v, v) <= 0
    # u(v, v') = v1'*x1(v') + v2'*x2(v') - p(v')
    # p(v') = v1'*x1(v') + v2'*x2(v') - sum_{t=0}^{v1'-1} x1(t,v2') - sum_{t=0}^{v2'-1} x2(v1',t)
    # So u(v, v') = sum_{t=0}^{v1'-1} x1(t,v2') + sum_{t=0}^{v2'-1} x2(v1',t)
    # u(v, v) = sum_{t=0}^{v1-1} x1(t,v2) + sum_{t=0}^{v2-1} x2(v1,t)
    
    # IC: sum_{t=0}^{v1'-1} x1(t,v2') + sum_{t=0}^{v2'-1} x2(v1',t) <= sum_{t=0}^{v1-1} x1(t,v2) + sum_{t=0}^{v2-1} x2(v1,t)
    
    # Rearranged: sum_{t=0}^{v1'-1} x1(t,v2') - sum_{t=0}^{v1-1} x1(t,v2) + sum_{t=0}^{v2'-1} x2(v1',t) - sum_{t=0}^{v2-1} x2(v1,t) <= 0
    
    A_rows = []
    b_vals = []
    
    for v_idx, (v1, v2) in enumerate(types):
        for vp_idx, (v1p, v2p) in enumerate(types):
            if v_idx == vp_idx:
                continue
            row = np.zeros(2 * n_types)
            # sum_{t=0}^{v1'-1} x1(t,v2')
            for t in range(v1p):
                t_idx = type_to_idx[(t, v2p)]
                row[t_idx] += 1  # x1 index
            # - sum_{t=0}^{v1-1} x1(t,v2)
            for t in range(v1):
                t_idx = type_to_idx[(t, v2)]
                row[t_idx] -= 1
            # sum_{t=0}^{v2'-1} x2(v1',t)
            for t in range(v2p):
                t_idx = type_to_idx[(v1p, t)]
                row[n_types + t_idx] += 1  # x2 index
            # - sum_{t=0}^{v2-1} x2(v1,t)
            for t in range(v2):
                t_idx = type_to_idx[(v1, t)]
                row[n_types + t_idx] -= 1
            A_rows.append(row)
            b_vals.append(0.0)
    
    # IR constraint: p(v) >= 0
    # p(v) = v1*x1(v) + v2*x2(v) - sum_{t=0}^{v1-1} x1(t,v2) - sum_{t=0}^{v2-1} x2(v1,t) >= 0
    # => -v1*x1(v) - v2*x2(v) + sum_{t=0}^{v1-1} x1(t,v2) + sum_{t=0}^{v2-1} x2(v1,t) <= 0
    
    for v_idx, (v1, v2) in enumerate(types):
        row = np.zeros(2 * n_types)
        row[v_idx] -= v1
        row[n_types + v_idx] -= v2
        for t in range(v1):
            t_idx = type_to_idx[(t, v2)]
            row[t_idx] += 1
        for t in range(v2):
            t_idx = type_to_idx[(v1, t)]
            row[n_types + t_idx] += 1
        A_rows.append(row)
        b_vals.append(0.0)
    
    A = np.array(A_rows) if A_rows else None
    b = np.array(b_vals) if b_vals else None
    
    # Bounds: 0 <= x1[v] <= 1, 0 <= x2[v] <= 1
    bounds = [(0, 1)] * (2 * n_types)
    
    # Solve LP
    if A is not None:
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    else:
        result = linprog(c, bounds=bounds, method='highs')
    
    if not result.success:
        # Fallback: uniform allocation
        allocs = np.ones((n_types, 2)) * 0.5
        # Compute payments
        pays = np.zeros(n_types)
        for v_idx, (v1, v2) in enumerate(types):
            pays[v_idx] = v1 * allocs[v_idx, 0] + v2 * allocs[v_idx, 1]
            for t in range(v1):
                t_idx = type_to_idx[(t, v2)]
                pays[v_idx] -= allocs[t_idx, 0]
            for t in range(v2):
                t_idx = type_to_idx[(v1, t)]
                pays[v_idx] -= allocs[t_idx, 1]
        return {'allocations': allocs, 'payments': pays}
    
    x = result.x
    x1 = x[:n_types]
    x2 = x[n_types:]
    
    # Clip to [0,1] for numerical safety
    x1 = np.clip(x1, 0, 1)
    x2 = np.clip(x2, 0, 1)
    
    # Compute payments
    payments = np.zeros(n_types)
    for v_idx, (v1, v2) in enumerate(types):
        payments[v_idx] = v1 * x1[v_idx] + v2 * x2[v_idx]
        for t in range(v1):
            t_idx = type_to_idx[(t, v2)]
            payments[v_idx] -= x1[t_idx]
        for t in range(v2):
            t_idx = type_to_idx[(v1, t)]
            payments[v_idx] -= x2[t_idx]
    
    # Ensure nonnegative payments (clip small negatives)
    payments = np.maximum(payments, 0)
    
    allocations = np.column_stack([x1, x2])
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
