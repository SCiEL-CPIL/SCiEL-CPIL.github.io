import numpy as np
from scipy.optimize import linprog
import time

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    # Parse problem structure
    values = [0, 1, 2, 3]
    
    # Get distribution
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
        # Ensure no negative probs
        probs = np.maximum(probs, 0)
        if probs.sum() == 0:
            probs = np.ones(16) / 16.0
        else:
            probs = probs / probs.sum()

    # Determine types
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
    
    n_types = len(types)
    assert n_types == 16, f"Expected 16 types, got {n_types}"
    
    type_to_vals = {idx: (v1, v2) for idx, (v1, v2) in enumerate(types)}
    type_vals = np.array([type_to_vals[i] for i in range(16)], dtype=float)
    v1_all = type_vals[:, 0]
    v2_all = type_vals[:, 1]
    
    # --- Reduced LP: 32 variables (x1[0..15], x2[16..31]) ---
    # Payments determined by allocations via IC + IR.
    # For a single additive bidder, IC implies:
    #   p(t) = v(t)·x(t) - U(t)
    # where U(t) is the utility at type t.
    # IC requires U(t) >= v(t)·x(t') - p(t') for all t'.
    # With IR: U(t) >= 0.
    # The minimal payment (max revenue) mechanism sets:
    #   U(t) = max(0, max_{t' != t} [v(t)·x(t') - v(t')·x(t') + U(t')])
    # This is recursive. Instead, we use the standard LP with payments as variables
    # but with a tighter formulation using the "pivot" structure.
    
    # Actually, let's use the direct approach: solve LP with 48 vars but with
    # better numerical settings and a verification/polish loop.
    
    n_vars = 48
    
    A_rows = []
    b_rows = []
    
    # IC: p(t) - p(t') <= v1(t)*(x1(t) - x1(t')) + v2(t)*(x2(t) - x2(t'))
    # => -v1(t)*x1(t) + v1(t)*x1(t') - v2(t)*x2(t) + v2(t)*x2(t') + p(t) - p(t') <= 0
    for t in range(16):
        for t2 in range(16):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            row[t] = -v1_all[t]
            row[t2] = v1_all[t]
            row[16 + t] = -v2_all[t]
            row[16 + t2] = v2_all[t]
            row[32 + t] = 1.0
            row[32 + t2] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR: p(t) <= v1(t)*x1(t) + v2(t)*x2(t)
    # => -v1(t)*x1(t) - v2(t)*x2(t) + p(t) <= 0
    for t in range(16):
        row = np.zeros(n_vars)
        row[t] = -v1_all[t]
        row[16 + t] = -v2_all[t]
        row[32 + t] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Objective: maximize sum p(t) * probs[t]
    c = np.zeros(n_vars)
    c[32:48] = -probs
    
    bounds = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16
    
    # Solve
    result = None
    try:
        options = {
            'presolve': True,
            'dual_feasibility_tolerance': 1e-18,
            'primal_feasibility_tolerance': 1e-18,
            'dual_objective_tolerance': 1e-18,
            'primal_objective_tolerance': 1e-18,
            'time_limit': 60
        }
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
    except Exception:
        pass

    if result is not None and result.success:
        x = result.x
        x1 = np.clip(x[0:16], 0, 1)
        x2 = np.clip(x[16:32], 0, 1)
        p = np.maximum(x[32:48], 0)
        allocations = np.column_stack((x1, x2))
        payments = p
    else:
        # Fallback: Zero mechanism
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
    
    # Verification and Minimal Fix
    def check_violations(allocs, pays):
        """Returns max IC violation and max IR violation."""
        max_ic_viol = 0.0
        max_ir_viol = 0.0
        for t in range(16):
            v1_t, v2_t = type_to_vals[t]
            # IR check
            util_t = v1_t * allocs[t, 0] + v2_t * allocs[t, 1] - pays[t]
            if util_t < 0:
                max_ir_viol = max(max_ir_viol, -util_t)
            
            # IC check: u(t) >= u(t') for all t'
            u_t = v1_t * allocs[t, 0] + v2_t * allocs[t, 1] - pays[t]
            for t2 in range(16):
                if t == t2:
                    continue
                u_t2_as_t = v1_t * allocs[t2, 0] + v2_t * allocs[t2, 1] - pays[t2]
                if u_t < u_t2_as_t:
                    viol = u_t2_as_t - u_t
                    if viol > max_ic_viol:
                        max_ic_viol = viol
        return max_ic_viol, max_ir_viol
    
    max_ic_viol, max_ir_viol = check_violations(allocations, payments)
    
    # If violations are within numerical noise, keep solution
    if max_ic_viol > 1e-9 or max_ir_viol > 1e-9:
        # Fix IR: ensure p(t) <= v1(t)*x1(t) + v2(t)*x2(t)
        for t in range(16):
            v1_t, v2_t = type_to_vals[t]
            max_pay = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
            if payments[t] > max_pay + 1e-12:
                payments[t] = max(0.0, max_pay)
        
        # Fix IC: for each violating pair (t, t'), increase payments[t] or decrease payments[t2]
        for _ in range(20):
            max_ic_viol, max_ir_viol = check_violations(allocations, payments)
            if max_ic_viol < 1e-10 and max_ir_viol < 1e-10:
                break
            
            if max_ic_viol > 1e-10:
                # Find the worst violating pair and adjust
                worst_viol = 0.0
                worst_t = -1
                worst_t2 = -1
                for t in range(16):
                    v1_t, v2_t = type_to_vals[t]
                    u_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1] - payments[t]
                    for t2 in range(16):
                        if t == t2:
                            continue
                        u_t2_as_t = v1_t * allocations[t2, 0] + v2_t * allocations[t2, 1] - payments[t2]
                        if u_t < u_t2_as_t:
                            viol = u_t2_as_t - u_t
                            if viol > worst_viol:
                                worst_viol = viol
                                worst_t = t
                                worst_t2 = t2
                
                if worst_t >= 0:
                    v1_w, v2_w = type_to_vals[worst_t]
                    required_max = v1_w * allocations[worst_t, 0] + v2_w * allocations[worst_t, 1] \
                                 - v1_w * allocations[worst_t2, 0] - v2_w * allocations[worst_t2, 1] + payments[worst_t2]
                    if required_max < payments[worst_t]:
                        payments[worst_t] = max(0.0, required_max)
            
            # Re-check IR after IC fixes
            for t in range(16):
                v1_t, v2_t = type_to_vals[t]
                max_pay = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
                if payments[t] > max_pay + 1e-12:
                    payments[t] = max(0.0, max_pay)
        
        # Final verification
        max_ic_viol, max_ir_viol = check_violations(allocations, payments)
        if max_ic_viol > 1e-8 or max_ir_viol > 1e-8:
            # Try a second LP solve with tighter constraints
            try:
                A_rows2 = []
                b_rows2 = []
                
                for t in range(16):
                    for t2 in range(16):
                        if t == t2:
                            continue
                        row = np.zeros(n_vars)
                        row[t] = -v1_all[t]
                        row[t2] = v1_all[t]
                        row[16 + t] = -v2_all[t]
                        row[16 + t2] = v2_all[t]
                        row[32 + t] = 1.0
                        row[32 + t2] = -1.0
                        A_rows2.append(row)
                        b_rows2.append(0.0)
                
                for t in range(16):
                    row = np.zeros(n_vars)
                    row[t] = -v1_all[t]
                    row[16 + t] = -v2_all[t]
                    row[32 + t] = 1.0
                    A_rows2.append(row)
                    b_rows2.append(0.0)
                
                A_ub2 = np.array(A_rows2)
                b_ub2 = np.array(b_rows2)
                
                c2 = np.zeros(n_vars)
                c2[32:48] = -probs
                
                bounds2 = [(0, 1)] * 16 + [(0, 1)] * 16 + [(0, None)] * 16
                
                options2 = {
                    'presolve': True,
                    'dual_feasibility_tolerance': 1e-16,
                    'primal_feasibility_tolerance': 1e-16,
                    'time_limit': 25
                }
                result2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, bounds=bounds2, method='highs', options=options2)
                
                if result2 is not None and result2.success:
                    x2 = result2.x
                    x1_2 = np.clip(x2[0:16], 0, 1)
                    x2_2 = np.clip(x2[16:32], 0, 1)
                    p2 = np.maximum(x2[32:48], 0)
                    allocs2 = np.column_stack((x1_2, x2_2))
                    pays2 = p2
                    
                    max_ic2, max_ir2 = check_violations(allocs2, pays2)
                    if max_ic2 < 1e-8 and max_ir2 < 1e-8:
                        allocations = allocs2
                        payments = pays2
            except Exception:
                pass
            
            # Final check
            max_ic_viol, max_ir_viol = check_violations(allocations, payments)
            if max_ic_viol > 1e-8 or max_ir_viol > 1e-8:
                allocations = np.zeros((16, 2))
                payments = np.zeros(16)
    
    # Deadline guard
    if time.time() > deadline:
        pass
    
    # High-precision polish: iteratively project onto feasible set with minimal changes
    def polish_solution(allocs, pays, max_iter=200):
        """Minimally adjust solution to ensure feasibility."""
        allocs = allocs.copy()
        pays = pays.copy()
        
        for _ in range(max_iter):
            ir_fixed = False
            ic_fixed = False
            
            for t in range(16):
                v1_t, v2_t = type_to_vals[t]
                max_pay = v1_t * allocs[t, 0] + v2_t * allocs[t, 1]
                if pays[t] > max_pay + 1e-15:
                    pays[t] = max(0.0, max_pay)
                    ir_fixed = True
            
            for t in range(16):
                v1_t, v2_t = type_to_vals[t]
                u_t = v1_t * allocs[t, 0] + v2_t * allocs[t, 1] - pays[t]
                for t2 in range(16):
                    if t == t2:
                        continue
                    u_t2_as_t = v1_t * allocs[t2, 0] + v2_t * allocs[t2, 1] - pays[t2]
                    if u_t < u_t2_as_t - 1e-15:
                        required = v1_t * allocs[t, 0] + v2_t * allocs[t, 1] - u_t2_as_t
                        if required > pays[t]:
                            max_pay_t = v1_t * allocs[t, 0] + v2_t * allocs[t, 1]
                            if required <= max_pay_t + 1e-15:
                                pays[t] = required
                                ic_fixed = True
                        else:
                            if pays[t2] > u_t2_as_t - u_t:
                                new_pay_t2 = max(0.0, pays[t2] - (u_t2_as_t - u_t + 1e-16))
                                v1_t2, v2_t2 = type_to_vals[t2]
                                max_pay_t2 = v1_t2 * allocs[t2, 0] + v2_t2 * allocs[t2, 1]
                                if new_pay_t2 <= max_pay_t2 + 1e-15:
                                    pays[t2] = new_pay_t2
                                    ic_fixed = True
            
            if not ir_fixed and not ic_fixed:
                break
        
        return allocs, pays
    
    allocations, payments = polish_solution(allocations, payments)
    
    # Final verification
    max_ic_viol, max_ir_viol = check_violations(allocations, payments)
    if max_ic_viol > 1e-8 or max_ir_viol > 1e-8:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
    
    return {'allocations': allocations, 'payments': payments}
