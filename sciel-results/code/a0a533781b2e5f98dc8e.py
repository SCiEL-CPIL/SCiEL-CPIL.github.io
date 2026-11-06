import numpy as np
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction.
    
    Strategy: Direct LP formulation with strict IC constraints (epsilon gap)
    to exploit the ability to lower payments while maintaining truthfulness.
    """
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin for 90s limit
    
    # 1. Parse Input
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            vals = list(distribution_spec.values())
            for v in vals:
                if isinstance(v, (list, np.ndarray)):
                    probs = np.array(v, dtype=float)
                    break
            else:
                probs = np.ones(16) / 16.0
    else:
        probs = np.array(distribution_spec, dtype=float)
        
    if probs.shape != (16,):
        probs = probs.flatten()
        if len(probs) != 16:
            if len(probs) > 16:
                probs = probs[:16]
            else:
                new_probs = np.zeros(16)
                new_probs[:len(probs)] = probs
                probs = new_probs

    total_p = np.sum(probs)
    if total_p > 1e-12:
        probs = probs / total_p
    else:
        probs = np.ones(16) / 16.0

    # 2. Define Grid
    grid_values = []
    for v1 in range(4):
        for v2 in range(4):
            grid_values.append((v1, v2))
    
    n_types = 16
    n_vars = 48  # 16 x1, 16 x2, 16 p

    # 3. Formulate LP
    # Maximize Revenue = sum(p_i * probs_i)
    # Minimize -sum(p_i * probs_i)
    c = np.zeros(n_vars)
    for i in range(16):
        c[32 + i] = -probs[i]
    
    A_rows = []
    b_rows = []
    
    # IR Constraints: v1*x1_i + v2*x2_i - p_i >= 0
    # => -v1*x1_i - v2*x2_i + p_i <= 0
    for i in range(16):
        v1, v2 = grid_values[i]
        row = np.zeros(n_vars)
        row[i] = -v1
        row[16 + i] = -v2
        row[32 + i] = 1
        A_rows.append(row)
        b_rows.append(0)
        
    # IC Constraints with Epsilon
    # u(i) >= u(j) + epsilon
    # v_i.x_i - p_i >= v_i.x_j - p_j + epsilon
    # -v_i.x_i + v_i.x_j + p_i - p_j <= -epsilon
    
    epsilon = 1e-9
    
    for i in range(16):
        v1_i, v2_i = grid_values[i]
        for j in range(16):
            if i == j:
                continue
            row = np.zeros(n_vars)
            row[i] = -v1_i
            row[16 + i] = -v2_i
            row[j] = v1_i
            row[16 + j] = v2_i
            row[32 + i] = 1
            row[32 + j] = -1
            
            A_rows.append(row)
            b_rows.append(-epsilon)
            
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds
    bounds = []
    for i in range(32):
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, None))
        
    # 4. Solve
    from scipy.optimize import linprog
    
    try:
        # Use tight tolerances
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-12, 
                               'primal_feasibility_tolerance': 1e-12})
        
        if time.time() - start_time > DEADLINE:
             # Fallback
             return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}

        if res.success:
            x = res.x
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            
            for i in range(16):
                allocations[i, 0] = x[i]
                allocations[i, 1] = x[16 + i]
                payments[i] = x[32 + i]
            
            # Clip to handle numerical noise
            allocations = np.clip(allocations, 0, 1)
            payments = np.clip(payments, 0, None)
            
            return {
                "allocations": allocations,
                "payments": payments
            }
        else:
            # If LP fails with epsilon, retry without epsilon (standard IC)
            # This handles cases where epsilon makes it infeasible
            b_rows_std = [0.0] * len(b_rows)
            # Rebuild b for standard IC (last 240 entries)
            # IR are first 16, IC are next 240
            b_rows_std[:16] = b_rows[:16]
            # IC part
            for k in range(16, len(b_rows)):
                b_rows_std[k] = 0.0
            
            res_std = linprog(c, A_ub=A_ub, b_ub=np.array(b_rows_std), bounds=bounds, method='highs', 
                              options={'presolve': True})
            if res_std.success:
                x = res_std.x
                allocations = np.zeros((16, 2))
                payments = np.zeros(16)
                for i in range(16):
                    allocations[i, 0] = x[i]
                    allocations[i, 1] = x[16 + i]
                    payments[i] = x[32 + i]
                allocations = np.clip(allocations, 0, 1)
                payments = np.clip(payments, 0, None)
                return {"allocations": allocations, "payments": payments}
            
            # Final fallback
            return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}
            
    except Exception:
        return {"allocations": np.zeros((16, 2)), "payments": np.zeros(16)}
