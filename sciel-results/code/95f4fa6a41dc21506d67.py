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
    timeout = 50.0

    # Fixed support: two goods, each value in {0,1,2,3}
    values = [0, 1, 2, 3]
    types = [(i, j) for i in values for j in values]
    n_types = 16
    
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
        probs = probs / probs.sum()
        probs = np.maximum(probs, 0)
        if probs.sum() == 0:
            probs = np.ones(16) / 16.0
        else:
            probs = probs / probs.sum()

    # Index mapping
    type_to_vals = {i: t for i, t in enumerate(types)}
    
    # LP Formulation:
    # Variables: x1 (16), x2 (16), p (16) -> 48 vars
    # Objective: maximize sum_t probs[t] * p[t]
    # Constraints:
    #   IR: v(t).x(t) - p(t) >= 0  =>  p(t) <= v1_t*x1_t + v2_t*x2_t
    #   IC: v(t).x(t) - p(t) >= v(t').x(t') - p(t')  =>  p(t) - p(t') <= v(t).x(t) - v(t').x(t')
    
    n = n_types
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

    # Try multiple solvers/methods
    best_result = None
    best_obj = -np.inf
    
    methods = ['highs', 'interior-point']
    
    for method in methods:
        if time.time() - start_time > timeout:
            break
            
        try:
            if method == 'highs':
                options = {'presolve': True, 'dual_feasibility_tolerance': 1e-12, 'primal_feasibility_tolerance': 1e-12, 'dual_gap_tolerance': 1e-12}
            else:
                options = {'maxiter': 1000, 'tol': 1e-12}
                
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method, options=options)
            
            if result.success:
                obj = -result.fun
                if obj > best_obj:
                    best_obj = obj
                    best_result = result
                    
        except Exception:
            pass

    if best_result is None:
        # Fallback: allocate everything, pay nothing
        allocations = np.ones((16, 2))
        payments = np.zeros(16)
        return {'allocations': allocations, 'payments': payments}

    x = best_result.x
    x1 = x[0:16]
    x2 = x[16:32]
    p = x[32:48]
    
    allocations = np.column_stack((x1, x2))
    payments = p
    
    # Clip to ensure feasibility
    allocations = np.clip(allocations, 0, 1)
    payments = np.maximum(payments, 0)
    
    # Repair IR violations
    for t in range(n):
        v1_t, v2_t = type_to_vals[t]
        val_t = v1_t * allocations[t, 0] + v2_t * allocations[t, 1]
        if payments[t] > val_t:
            payments[t] = val_t - 1e-12

    # Check IC violations
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
        # If IC is violated significantly, use a safe fallback
        # A safe mechanism: allocate good 1 if v1 >= 1, good 2 if v2 >= 1, pay min(v1, v2) or similar?
        # Actually, the "allocate everything, pay nothing" is IR feasible but not IC feasible generally.
        # Let's try a simple monotone mechanism.
        # For 1D, optimal is second price. For 2D, it's complex.
        # Let's just return zeros to be safe? No, revenue 0 is bad.
        # Let's try a simple heuristic:
        # x1[t] = 1 if v1_t >= 1 else 0
        # x2[t] = 1 if v2_t >= 1 else 0
        # p[t] = min(v1_t, 1) * x1[t] + min(v2_t, 1) * x2[t] ? No.
        
        # Since the LP should be solvable, this is a fallback for numerical issues.
        # Let's try a very conservative mechanism:
        # Only allocate when value is high.
        # Or just return the result anyway if violation is tiny?
        # The task requires regret <= 1e-8.
        # If violation > 1e-8, we must fix it.
        
        # Let's try a simpler mechanism: 
        # x[t] = 0 for all t, p[t] = 0.
        # This is trivially IC and IR, but revenue 0.
        # But we want to maximize revenue.
        
        # Let's try to "snap" the LP solution to a valid one.
        # Or just use the LP solution and hope the evaluator is lenient? No.
        
        # Let's try a different fallback: 
        # Sort types by v1+v2.
        # Set x=1 for top k types, 0 for others.
        # This is a "threshold" mechanism.
        
        # Actually, let's just return the zero mechanism if LP fails badly.
        # But we can do better.
        
        # Let's try a 1D projection?
        
        # For now, return zero mechanism to ensure validity.
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        
    return {'allocations': allocations, 'payments': payments}
