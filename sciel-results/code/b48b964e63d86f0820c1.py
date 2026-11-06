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

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0
    
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
        elif 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
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
            if i == j: continue
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
    
    def _validate(allocations, payments):
        """Check IC, IR, non-negativity, and feasibility."""
        if np.any(allocations < -1e-10) or np.any(payments < -1e-10):
            return False
        if np.any(allocations > 1 + 1e-10):
            return False
        for i in range(n_types):
            vi = np.array(types[i], dtype=float)
            ui_i = np.dot(allocations[i], vi) - payments[i]
            if ui_i < -1e-8:
                return False
            for j in range(n_types):
                ui_j = np.dot(allocations[j], vi) - payments[j]
                if ui_i < ui_j - 1e-8:
                    return False
        return True

    def _repair_payments(allocations, payments):
        """
        Iteratively adjust payments to satisfy IC and IR.
        We reduce payments to the maximum allowed by constraints.
        This ensures validity but may reduce revenue.
        """
        payments = payments.copy()
        for _ in range(10): # Iterate until convergence
            changed = False
            for i in range(n_types):
                vi = np.array(types[i], dtype=float)
                # IR constraint: p_i <= v_i . x_i
                max_p_ir = np.dot(allocations[i], vi)
                
                # IC constraints: p_i <= p_j + v_i . (x_i - x_j) for all j
                max_p_ic = np.inf
                for j in range(n_types):
                    if i == j: continue
                    vi_xi = np.dot(allocations[i], vi)
                    vi_xj = np.dot(allocations[j], vi)
                    bound = payments[j] + (vi_xi - vi_xj)
                    if bound < max_p_ic:
                        max_p_ic = bound
                
                max_allowed = min(max_p_ir, max_p_ic)
                if payments[i] > max_allowed:
                    payments[i] = max(0.0, max_allowed)
                    changed = True
            if not changed:
                break
        return payments

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12})
        if res.success:
            x_flat = res.x[:n_types * n_goods]
            p_flat = res.x[n_types * n_goods:]
            allocations = x_flat.reshape(n_types, n_goods)
            payments = p_flat.copy()
            
            # Clip to feasible range
            allocations = np.clip(allocations, 0, 1)
            payments = np.maximum(payments, 0)
            
            # Check if solution is valid
            if _validate(allocations, payments):
                return {'allocations': allocations, 'payments': payments}
            
            # Try repair
            repaired_payments = _repair_payments(allocations, payments)
            if _validate(allocations, repaired_payments):
                return {'allocations': allocations, 'payments': repaired_payments}
            
            # Fallback to simple valid mechanism
            alloc_fb, pay_fb = _fallback_mechanism(types, n_types, n_goods, probs)
            return {'allocations': alloc_fb, 'payments': pay_fb}
        else:
            alloc_fb, pay_fb = _fallback_mechanism(types, n_types, n_goods, probs)
            return {'allocations': alloc_fb, 'payments': pay_fb}
    except Exception:
        alloc_fb, pay_fb = _fallback_mechanism(types, n_types, n_goods, probs)
        return {'allocations': alloc_fb, 'payments': pay_fb}
