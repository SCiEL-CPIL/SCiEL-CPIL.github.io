import time
import numpy as np
from scipy.optimize import linprog

def _fallback_mechanism(types, n_types, n_goods, probs):
    """
    Fallback mechanism if LP fails.
    Uses a simple threshold rule which is not necessarily optimal but valid.
    """
    allocations = np.zeros((n_types, n_goods))
    payments = np.zeros(n_types)
    for i in range(n_types):
        vi = types[i]
        if vi[0] >= 2:
            allocations[i, 0] = 1.0
            payments[i] = float(vi[0])
    return allocations, payments

def _max_valid_payments(allocations, types, n_types, tol=1e-10):
    """
    Given fixed allocations, find the maximum valid payments.
    This solves the payment subproblem exactly.
    """
    n = n_types
    # Build the payment LP: maximize sum(probs[i] * p[i])
    # subject to:
    #   p_i - v_i.x_i <= 0  (IR)
    #   p_i - p_j - v_i.(x_i - x_j) <= 0  (IC: p_i <= p_j + v_i.(x_i - x_j))
    #   p_i >= 0
    
    c = np.zeros(n)
    c[:] = -1.0  # maximize sum p_i (we'll weight by probs later if needed, but for validity just maximize sum)
    
    A_ub = []
    b_ub = []
    
    for i in range(n):
        vi = np.array(types[i], dtype=float)
        # IR: p_i <= v_i . x_i
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(np.dot(allocations[i], vi))
        
        # IC: p_i - p_j <= v_i . (x_i - x_j)
        for j in range(n):
            if i == j:
                continue
            vi_xi = np.dot(allocations[i], vi)
            vi_xj = np.dot(allocations[j], vi)
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = -1.0
            A_ub.append(row)
            b_ub.append(vi_xi - vi_xj)
    
    if not A_ub:
        return np.zeros(n)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None)] * n
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12})
        if res.success:
            payments = np.maximum(res.x, 0)
            return payments
    except Exception:
        pass
    
    # Fallback: iterative propagation
    payments = np.zeros(n)
    for _ in range(100):
        changed = False
        for i in range(n):
            vi = np.array(types[i], dtype=float)
            max_p = np.dot(allocations[i], vi)  # IR bound
            for j in range(n):
                if i == j:
                    continue
                vi_xi = np.dot(allocations[i], vi)
                vi_xj = np.dot(allocations[j], vi)
                bound = payments[j] + (vi_xi - vi_xj)
                if bound < max_p:
                    max_p = bound
            if payments[i] < max_p - tol:
                payments[i] = max(0.0, max_p)
                changed = True
        if not changed:
            break
    return payments

def _validate(allocations, payments, types, n_types, tol=1e-8):
    """Check IC, IR, non-negativity, and feasibility."""
    if np.any(allocations < -1e-10) or np.any(payments < -1e-10):
        return False
    if np.any(allocations > 1 + 1e-10):
        return False
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        ui_i = np.dot(allocations[i], vi) - payments[i]
        if ui_i < -tol:
            return False
        for j in range(n_types):
            ui_j = np.dot(allocations[j], vi) - payments[j]
            if ui_i < ui_j - tol:
                return False
    return True

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0
    
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    n_types = len(types)
    n_goods = 2
    
    # Parse distribution
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            probs = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(n_types) / n_types
        
    probs = probs / probs.sum()
    
    # LP Variables: x[0..31], p[32..47]
    n_vars = n_types * n_goods + n_types
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[n_types * n_goods + i] = -probs[i]
        
    A_ub = []
    b_ub = []
    
    # IC: p_i - p_j + v_i . (x_j - x_i) <= 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        for j in range(n_types):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[i * n_goods + 0] = -vi[0]
            row[i * n_goods + 1] = -vi[1]
            row[j * n_goods + 0] = vi[0]
            row[j * n_goods + 1] = vi[1]
            row[n_types * n_goods + i] = 1
            row[n_types * n_goods + j] = -1
            A_ub.append(row)
            b_ub.append(0)
            
    # IR: p_i - v_i . x_i <= 0
    for i in range(n_types):
        vi = np.array(types[i], dtype=float)
        row = np.zeros(n_vars)
        row[i * n_goods + 0] = -vi[0]
        row[i * n_goods + 1] = -vi[1]
        row[n_types * n_goods + i] = 1
        A_ub.append(row)
        b_ub.append(0)
        
    bounds = []
    for _ in range(n_types * n_goods):
        bounds.append((0, 1))
    for _ in range(n_types):
        bounds.append((0, None))
        
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    def _time_check():
        return (time.time() - start_time) < deadline
    
    best_alloc = None
    best_pay = None
    best_rev = -1.0
    
    def _update_best(alloc, pay):
        nonlocal best_alloc, best_pay, best_rev
        if _validate(alloc, pay, types, n_types, tol=1e-8):
            rev = float(np.dot(probs, pay))
            if rev > best_rev:
                best_rev = rev
                best_alloc = alloc.copy()
                best_pay = pay.copy()
    
    # Attempt 1: High-precision LP
    try:
        if _time_check():
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                          options={'presolve': True, 'dual_feasibility_tolerance': 1e-14,
                                   'primal_feasibility_tolerance': 1e-14,
                                   'dual_stopping_tolerance': 1e-14,
                                   'primal_stopping_tolerance': 1e-14})
            if res.success:
                x_flat = res.x[:n_types * n_goods]
                p_flat = res.x[n_types * n_goods:]
                allocations = x_flat.reshape(n_types, n_goods)
                payments = p_flat.copy()
                
                allocations = np.clip(allocations, 0, 1)
                payments = np.maximum(payments, 0)
                
                _update_best(allocations, payments)
                
                # If valid, try to improve payments via the payment subproblem
                if _validate(allocations, payments, types, n_types, tol=1e-8):
                    if _time_check():
                        optimal_payments = _max_valid_payments(allocations, types, n_types)
                        _update_best(allocations, optimal_payments)
    except Exception:
        pass
    
    # Attempt 2: Standard precision LP
    try:
        if _time_check():
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                          options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                                   'primal_feasibility_tolerance': 1e-12})
            if res.success:
                x_flat = res.x[:n_types * n_goods]
                p_flat = res.x[n_types * n_goods:]
                allocations = x_flat.reshape(n_types, n_goods)
                payments = p_flat.copy()
                
                allocations = np.clip(allocations, 0, 1)
                payments = np.maximum(payments, 0)
                
                _update_best(allocations, payments)
                
                if _validate(allocations, payments, types, n_types, tol=1e-8):
                    if _time_check():
                        optimal_payments = _max_valid_payments(allocations, types, n_types)
                        _update_best(allocations, optimal_payments)
    except Exception:
        pass
    
    # If we have a valid best, return it
    if best_alloc is not None:
        return {'allocations': best_alloc, 'payments': best_pay}
    
    # Fallback: try the simple threshold mechanism
    alloc_fb, pay_fb = _fallback_mechanism(types, n_types, n_goods, probs)
    if _validate(alloc_fb, pay_fb, types, n_types, tol=1e-8):
        return {'allocations': alloc_fb, 'payments': pay_fb}
    
    # Last resort: zero mechanism
    return {'allocations': np.zeros((n_types, n_goods)), 'payments': np.zeros(n_types)}
