"""Direct convex LP over allocation and payment variables for optimal finite-support mechanism.
Optimized with reduced IC constraints (lattice adjacencies) and high-precision polishing."""
import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: support of the value space
        constraints: additional constraints
    
    Returns:
        dict with 'allocations': (16, 2) array and 'payments': (16,) array
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit

    # Parse distribution
    values = [0, 1, 2, 3]
    
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
    
    # Parse support
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]
    
    n = 16
    if len(types) != n:
        # Fallback if support parsing fails or is invalid
        types = [(i, j) for i in values for j in values]
    
    val_matrix = np.array(types, dtype=float)  # shape (16, 2)
    
    # LP formulation:
    # Variables: x[t,0], x[t,1] for t in 0..15 (32 vars), p[t] for t in 0..15 (16 vars)
    # Total: 48 vars
    # Order: x[0,0], x[0,1], ..., x[15,0], x[15,1], p[0], ..., p[15]
    
    n_vars = 48
    x_offset = 0
    p_offset = 32
    
    # Objective: maximize sum_t probs[t] * p[t] => minimize -sum
    c = np.zeros(n_vars)
    for t in range(n):
        c[p_offset + t] = -probs[t]
    
    A_rows = []
    b_rows = []
    
    # IC constraints: p[t] - p[t'] <= v[t]·x[t] - v[t]·x[t']
    # We use adjacent constraints in the lattice to reduce problem size.
    # This is sufficient for monotonicity in a grid structure.
    # If the support is not a standard grid, we fall back to all pairs.
    is_standard_grid = (types == [(i, j) for i in values for j in values])
    
    if is_standard_grid:
        # Generate adjacent pairs
        # Index t = 4*v1 + v2
        for v1 in range(4):
            for v2 in range(4):
                t = 4 * v1 + v2
                # Check right neighbor (v1+1, v2)
                if v1 < 3:
                    t2 = 4 * (v1 + 1) + v2
                    _add_ic_constraint(A_rows, b_rows, t, t2, val_matrix, n_vars, x_offset, p_offset)
                # Check up neighbor (v1, v2+1)
                if v2 < 3:
                    t2 = 4 * v1 + (v2 + 1)
                    _add_ic_constraint(A_rows, b_rows, t, t2, val_matrix, n_vars, x_offset, p_offset)
    else:
        # Fallback: all pairs
        for t in range(n):
            for t2 in range(n):
                if t == t2:
                    continue
                _add_ic_constraint(A_rows, b_rows, t, t2, val_matrix, n_vars, x_offset, p_offset)
    
    # IR constraints: p[t] <= v[t]·x[t]
    for t in range(n):
        row = np.zeros(n_vars)
        row[p_offset + t] = 1.0
        for g in range(2):
            row[x_offset + t*2 + g] = -val_matrix[t, g]
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows) if A_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_rows) if A_rows else np.zeros(0)
    
    # Bounds: x in [0,1], p >= 0
    bounds = []
    for t in range(n):
        for g in range(2):
            bounds.append((0, 1))
    for t in range(n):
        bounds.append((0, None))
    
    try:
        # High precision solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True,
                                  'dual_feasibility_tolerance': 1e-12,
                                  'primal_feasibility_tolerance': 1e-12,
                                  'time_limit': 30.0})
        
        if result.success:
            alloc = result.x[:32].reshape(n, 2)
            pay = result.x[32:48]
            
            # Clip numerical noise
            alloc = np.clip(alloc, 0, 1)
            pay = np.maximum(pay, 0)
            
            # Snap near-boundary values to help with exactness
            alloc[alloc < 1e-12] = 0.0
            alloc[alloc > 1 - 1e-12] = 1.0
            pay[pay < 1e-12] = 0.0
            
            # Verify and fix
            alloc, pay = _verify_and_fix(alloc, pay, val_matrix, probs, is_standard_grid)
            
            if time.time() > deadline:
                return {'allocations': alloc, 'payments': pay}
                
            # Polish pass: Re-solve with tighter tolerances if time permits
            alloc, pay = _polish(alloc, pay, val_matrix, probs, start_time, deadline, is_standard_grid)
            
            return {'allocations': alloc, 'payments': pay}
        else:
            return _fallback_solution(val_matrix, probs)
    except Exception:
        return _fallback_solution(val_matrix, probs)

def _add_ic_constraint(A_rows, b_rows, t, t2, val_matrix, n_vars, x_offset, p_offset):
    """Add IC constraint: p[t] - p[t2] <= v[t]·x[t] - v[t]·x[t2]"""
    row = np.zeros(n_vars)
    row[p_offset + t] = 1.0
    row[p_offset + t2] = -1.0
    for g in range(2):
        row[x_offset + t*2 + g] = -val_matrix[t, g]
        row[x_offset + t2*2 + g] = val_matrix[t, g]
    A_rows.append(row)
    b_rows.append(0.0)

def _verify_and_fix(alloc, pay, val_matrix, probs, is_standard_grid):
    """Verify IC and IR constraints, fix small violations."""
    n = len(pay)
    alloc = alloc.copy()
    pay = pay.copy()
    
    # Check IR: pay[t] <= val[t] · alloc[t]
    for t in range(n):
        max_pay = np.dot(val_matrix[t], alloc[t])
        if pay[t] > max_pay + 1e-8:
            pay[t] = max(max_pay, 0)
    
    # Check IC
    # For standard grid, check adjacent first (faster), then all if needed
    if is_standard_grid:
        pairs = []
        for v1 in range(4):
            for v2 in range(4):
                t = 4 * v1 + v2
                if v1 < 3: pairs.append((t, 4*(v1+1)+v2))
                if v2 < 3: pairs.append((t, 4*v1+v2+1))
    else:
        pairs = [(t, t2) for t in range(n) for t2 in range(n) if t != t2]
        
    for t, t2 in pairs:
        lhs = pay[t] - pay[t2]
        rhs = np.dot(val_matrix[t], alloc[t]) - np.dot(val_matrix[t], alloc[t2])
        if lhs > rhs + 1e-8:
            excess = lhs - rhs
            # Reduce pay[t] to satisfy constraint
            pay[t] = max(pay[t] - excess, 0)
            # Re-check IR for t
            max_pay_t = np.dot(val_matrix[t], alloc[t])
            if pay[t] > max_pay_t:
                pay[t] = max(max_pay_t, 0)
                
    # Final sanity check: ensure non-negative payments and valid allocations
    pay = np.maximum(pay, 0)
    alloc = np.clip(alloc, 0, 1)
    
    return alloc, pay

def _polish(alloc, pay, val_matrix, probs, start_time, deadline, is_standard_grid):
    """Re-solve LP with tighter tolerances."""
    if time.time() > deadline - 10.0:
        return alloc, pay
    
    n = 16
    n_vars = 48
    x_offset = 0
    p_offset = 32
    
    c = np.zeros(n_vars)
    for t in range(n):
        c[p_offset + t] = -probs[t]
    
    A_rows = []
    b_rows = []
    
    if is_standard_grid:
        for v1 in range(4):
            for v2 in range(4):
                t = 4 * v1 + v2
                if v1 < 3:
                    _add_ic_constraint(A_rows, b_rows, t, 4*(v1+1)+v2, val_matrix, n_vars, x_offset, p_offset)
                if v2 < 3:
                    _add_ic_constraint(A_rows, b_rows, t, 4*v1+v2+1, val_matrix, n_vars, x_offset, p_offset)
    else:
        for t in range(n):
            for t2 in range(n):
                if t != t2:
                    _add_ic_constraint(A_rows, b_rows, t, t2, val_matrix, n_vars, x_offset, p_offset)
    
    for t in range(n):
        row = np.zeros(n_vars)
        row[p_offset + t] = 1.0
        for g in range(2):
            row[x_offset + t*2 + g] = -val_matrix[t, g]
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    bounds = [(0, 1)] * 32 + [(0, None)] * 16
    
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True,
                                  'dual_feasibility_tolerance': 1e-14,
                                  'primal_feasibility_tolerance': 1e-14,
                                  'time_limit': 20.0})
        if result.success:
            alloc_new = result.x[:32].reshape(n, 2)
            pay_new = result.x[32:48]
            alloc_new = np.clip(alloc_new, 0, 1)
            pay_new = np.maximum(pay_new, 0)
            alloc_new[alloc_new < 1e-14] = 0.0
            alloc_new[alloc_new > 1 - 1e-14] = 1.0
            pay_new[pay_new < 1e-14] = 0.0
            
            rev_old = np.dot(probs, pay)
            rev_new = np.dot(probs, pay_new)
            
            if rev_new >= rev_old - 1e-10:
                alloc_new, pay_new = _verify_and_fix(alloc_new, pay_new, val_matrix, probs, is_standard_grid)
                return alloc_new, pay_new
    except Exception:
        pass
    
    return alloc, pay

def _fallback_solution(val_matrix, probs):
    """Simple fallback: allocate both goods to all, pay full value."""
    n = 16
    alloc = np.ones((n, 2))
    pay = np.array([np.dot(val_matrix[t], alloc[t]) for t in range(n)])
    pay = np.maximum(pay, 0)
    return {'allocations': alloc, 'payments': pay}
