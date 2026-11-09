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
    timeout = 50.0  # Safety margin before 60s limit

    # Fixed support: two goods, each value in {0,1,2,3}
    values = [0, 1, 2, 3]
    types = [(i, j) for i in values for j in values]
    n_types = 16
    n = n_types

    # Get distribution
    probs = None
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            elif 'values' in distribution_spec and 'probs' in distribution_spec:
                vals = np.array(distribution_spec['values'])
                ps = np.array(distribution_spec['probs'])
                if len(vals) == 16 and len(ps) == 16:
                    probs = ps
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = np.maximum(probs, 0)
        if probs.sum() == 0:
            probs = np.ones(16) / 16.0
        else:
            probs = probs / probs.sum()

    # Index mapping
    type_to_vals = {i: t for i, t in enumerate(types)}

    # Primal LP:
    # Variables: x1 (16), x2 (16), p (16) -> 48 vars
    # Maximize: sum_t probs[t] * p[t]
    # Subject to:
    #   IC: p(t) - p(t') <= v(t).x(t) - v(t').x(t')  for all t != t'
    #   IR: p(t) <= v(t).x(t)  for all t
    #   0 <= x1,t <= 1, 0 <= x2,t <= 1, p(t) >= 0

    c = np.zeros(3 * n)
    for t in range(n):
        c[32 + t] = -probs[t]

    A_rows = []
    b_rows = []
    
    # IC constraints: p(t) - p(t') <= v(t).x(t) - v(t').x(t')
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        for t2 in range(n):
            if t == t2:
                continue
            row = np.zeros(3 * n)
            row[0 + t] = -v1_t
            row[16 + t] = -v2_t
            row[0 + t2] = v1_t
            row[16 + t2] = v2_t
            row[32 + t] = 1
            row[32 + t2] = -1
            A_rows.append(row)
            b_rows.append(0)
            
    # IR constraints: p(t) <= v1_t*x1_t + v2_t*x2_t
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        row = np.zeros(3 * n)
        row[0 + t] = -v1_t
        row[16 + t] = -v2_t
        row[32 + t] = 1
        A_rows.append(row)
        b_rows.append(0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    bounds = []
    for _ in range(n):
        bounds.append((0, 1))
    for _ in range(n):
        bounds.append((0, 1))
    for _ in range(n):
        bounds.append((0, None))

    # Solve primal LP with high precision
    best_result = None
    best_obj = -np.inf
    
    methods = ['highs', 'interior-point']
    
    for method in methods:
        if time.time() - start_time > timeout:
            break
            
        try:
            if method == 'highs':
                options = {'presolve': True, 'dual_feasibility_tolerance': 1e-14, 
                          'primal_feasibility_tolerance': 1e-14, 'dual_gap_tolerance': 1e-14}
            else:
                options = {'maxiter': 2000, 'tol': 1e-14}
                
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method, options=options)
            
            if result.success:
                obj = -result.fun
                if obj > best_obj:
                    best_obj = obj
                    best_result = result
                    
        except Exception:
            pass

    if best_result is None:
        # Fallback: zero mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}

    x = best_result.x
    x1 = x[0:16].copy()
    x2 = x[16:32].copy()
    p = x[32:48].copy()
    
    allocations = np.column_stack((x1, x2))
    payments = p
    
    # Clip to ensure feasibility
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Verification and repair: ensure IC and IR to 1e-8 precision
    # IR: p(t) <= v(t).x(t)
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        val_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        if payments[t] > val_t + 1e-12:
            payments[t] = max(0, val_t - 1e-12)
    
    # IC verification: p(t) - p(t') <= v(t).x(t) - v(t').x(t')
    # If violated, we need to adjust. The safest approach is to lower p(t) or raise p(t')
    # but raising p(t') might violate IR. So we lower p(t) if needed.
    # However, lowering p(t) might violate IC for other pairs where t is the "report" type.
    # A more robust approach: solve a small correction LP if violations exist.
    
    max_violation = 0
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        for t2 in range(n):
            if t == t2:
                continue
            v1_t2, v2_t2 = type_to_vals[t2]
            lhs = payments[t] - payments[t2]
            rhs = v1_t * (allocations[t, 0] - allocations[t2, 0]) + v2_t * (allocations[t, 1] - allocations[t2, 1])
            if lhs > rhs:
                max_violation = max(max_violation, lhs - rhs)
    
    if max_violation > 1e-8:
        # Solve a correction LP to find minimal adjustments to payments
        # that restore IC while maintaining IR and non-negativity
        # Variables: delta_p (16), minimize sum |delta_p[t]|
        # Constraints:
        #   (p[t] + delta_p[t]) - (p[t2] + delta_p[t2]) <= v(t).x(t) - v(t').x(t')
        #   0 <= p[t] + delta_p[t] <= v(t).x(t)
        
        if time.time() - start_time < timeout - 5:
            try:
                # Minimize sum of absolute deviations: use aux variables
                # Variables: delta_p (16), pos (16), neg (16) -> 48 vars
                # delta_p[t] = pos[t] - neg[t]
                # Minimize sum(pos + neg)
                
                c_corr = np.zeros(3 * n)
                for t in range(n):
                    c_corr[16 + t] = 1  # pos
                    c_corr[32 + t] = 1  # neg
                
                A_corr = []
                b_corr = []
                
                # IC constraints with adjusted payments
                for t in range(n):
                    v1_t, v2_t = type_to_vals[t]
                    for t2 in range(n):
                        if t == t2:
                            continue
                        v1_t2, v2_t2 = type_to_vals[t2]
                        row = np.zeros(3 * n)
                        # (p[t] + pos[t] - neg[t]) - (p[t2] + pos[t2] - neg[t2]) <= v(t).x(t) - v(t').x(t')
                        # pos[t] - neg[t] - pos[t2] + neg[t2] <= v(t).x(t) - v(t').x(t') - p[t] + p[t2]
                        row[16 + t] = 1
                        row[32 + t] = -1
                        row[16 + t2] = -1
                        row[32 + t2] = 1
                        rhs_val = v1_t * (allocations[t, 0] - allocations[t2, 0]) + \
                                  v2_t * (allocations[t, 1] - allocations[t2, 1]) - payments[t] + payments[t2]
                        A_corr.append(row)
                        b_corr.append(rhs_val)
                
                # IR: 0 <= p[t] + delta_p[t] <= v(t).x(t)
                # p[t] + pos[t] - neg[t] >= 0  =>  -pos[t] + neg[t] <= p[t]
                for t in range(n):
                    row = np.zeros(3 * n)
                    row[16 + t] = -1
                    row[32 + t] = 1
                    A_corr.append(row)
                    b_corr.append(payments[t])
                
                # p[t] + pos[t] - neg[t] <= v(t).x(t)  =>  pos[t] - neg[t] <= v(t).x(t) - p[t]
                for t in range(n):
                    v1_t, v2_t = type_to_vals[t]
                    val_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
                    row = np.zeros(3 * n)
                    row[16 + t] = 1
                    row[32 + t] = -1
                    A_corr.append(row)
                    b_corr.append(val_t - payments[t])
                
                A_corr = np.array(A_corr)
                b_corr = np.array(b_corr)
                
                bounds_corr = [(0, None)] * (3 * n)
                
                result_corr = linprog(c_corr, A_ub=A_corr, b_ub=b_corr, bounds=bounds_corr, 
                                     method='highs', options={'presolve': True,
                                                          'dual_feasibility_tolerance': 1e-12,
                                                          'primal_feasibility_tolerance': 1e-12,
                                                          'dual_gap_tolerance': 1e-12})
                
                if result_corr.success:
                    delta_p = result_corr.x[0:16]
                    new_p = payments + delta_p
                    new_p = np.clip(new_p, 0, None)
                    
                    # Verify the new solution
                    ir_ok = True
                    ic_ok = True
                    for t in range(n):
                        v1_t, v2_t = type_to_vals[t]
                        val_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
                        if new_p[t] > val_t + 1e-10:
                            ir_ok = False
                        for t2 in range(n):
                            if t == t2:
                                continue
                            v1_t2, v2_t2 = type_to_vals[t2]
                            lhs = new_p[t] - new_p[t2]
                            rhs = v1_t * (allocations[t, 0] - allocations[t2, 0]) + \
                                  v2_t * (allocations[t, 1] - allocations[t2, 1])
                            if lhs > rhs + 1e-10:
                                ic_ok = False
                    
                    if ir_ok and ic_ok:
                        payments = new_p
                        # Revenue decreased, but we have a valid mechanism
                        # The objective value is now sum(probs[t] * payments[t])
                    
            except Exception:
                pass
    
    # Final safety: if still invalid, use zero mechanism
    max_violation_final = 0
    ir_violation = 0
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        val_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        if payments[t] > val_t:
            ir_violation = max(ir_violation, payments[t] - val_t)
        for t2 in range(n):
            if t == t2:
                continue
            v1_t2, v2_t2 = type_to_vals[t2]
            lhs = payments[t] - payments[t2]
            rhs = v1_t * (allocations[t, 0] - allocations[t2, 0]) + v2_t * (allocations[t, 1] - allocations[t2, 1])
            if lhs > rhs:
                max_violation_final = max(max_violation_final, lhs - rhs)
    
    if max_violation_final > 1e-8 or ir_violation > 1e-8:
        # Last resort: zero mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
    
    return {'allocations': allocations, 'payments': payments}
