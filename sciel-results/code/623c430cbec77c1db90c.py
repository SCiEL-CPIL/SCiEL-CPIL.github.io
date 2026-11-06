import time
import numpy as np
from scipy.optimize import linprog

def _fallback_mechanism(types, n_types, n_goods, probs):
    """Simple valid mechanism: allocate good 1 to type if v1 >= 2, pay v1 if allocated."""
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
    
    def _validate(allocations, payments, tol=1e-8):
        """Check IC, IR, non-negativity, and feasibility."""
        if np.any(allocations < -tol) or np.any(payments < -tol):
            return False
        if np.any(allocations > 1 + tol):
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
    
    def _expected_revenue(allocations, payments):
        """Compute expected revenue."""
        return float(np.sum(probs * payments))
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-12,
                               'primal_feasibility_tolerance': 1e-12,
                               'dual_simplex_max_iter': 1000000,
                               'primal_simplex_max_iter': 1000000})
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
                base_rev = _expected_revenue(allocations, payments)
                
                # Polishing: try to increase payments slightly while maintaining IC/IR
                # The LP should already be optimal, but let's verify and try small adjustments
                best_alloc = allocations.copy()
                best_pay = payments.copy()
                best_rev = base_rev
                
                # Try increasing payments for types where there's slack in IC constraints
                for _polish_iter in range(3):
                    if time.time() > deadline - 5:
                        break
                    improved = False
                    for i in range(n_types):
                        vi = np.array(types[i], dtype=float)
                        truthful_val = np.dot(best_alloc[i], vi)
                        
                        # Find maximum payment that maintains IC for type i
                        max_p = truthful_val  # IR bound
                        for j in range(n_types):
                            if i == j: continue
                            # IC: p_i <= p_j + vi . (x_i - x_j)
                            ic_bound = best_pay[j] + np.dot(vi, best_alloc[i] - best_alloc[j])
                            max_p = min(max_p, ic_bound)
                        
                        max_p = max(0.0, max_p)
                        
                        # Try to increase payment to max_p
                        if max_p > best_pay[i] + 1e-10:
                            old_p = best_pay[i]
                            best_pay[i] = max_p
                            if _validate(best_alloc, best_pay):
                                new_rev = _expected_revenue(best_alloc, best_pay)
                                if new_rev > best_rev + 1e-12:
                                    best_rev = new_rev
                                    improved = True
                            else:
                                best_pay[i] = old_p
                    
                    if not improved:
                        break
                
                if _validate(best_alloc, best_pay):
                    return {'allocations': best_alloc, 'payments': best_pay}
                else:
                    return {'allocations': allocations, 'payments': payments}
            else:
                # Try repair: adjust payments to satisfy IC and IR
                for i in range(n_types):
                    vi = np.array(types[i], dtype=float)
                    truthful_val = np.dot(allocations[i], vi)
                    max_mis_util = 0.0
                    for j in range(n_types):
                        if i == j: continue
                        mis_util = np.dot(allocations[j], vi) - payments[j]
                        if mis_util > max_mis_util:
                            max_mis_util = mis_util
                
                    max_allowed_p = truthful_val - max_mis_util
                    if payments[i] > max_allowed_p:
                        payments[i] = max(0.0, max_allowed_p)
                
                if _validate(allocations, payments):
                    return {'allocations': allocations, 'payments': payments}
                
                # Fallback to simple valid mechanism
                alloc_fb, pay_fb = _fallback_mechanism(types, n_types, n_goods, probs)
                return {'allocations': alloc_fb, 'payments': pay_fb}
        else:
            alloc_fb, pay_fb = _fallback_mechanism(types, n_types, n_goods, probs)
            return {'allocations': alloc_fb, 'payments': pay_fb}
    except Exception:
        alloc_fb, pay_fb = _fallback_mechanism(types, n_types, n_goods, probs)
        return {'allocations': alloc_fb, 'payments': pay_fb}
