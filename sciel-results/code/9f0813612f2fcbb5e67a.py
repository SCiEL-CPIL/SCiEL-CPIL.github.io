import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t_start = time.time()
    deadline = t_start + 85.0  # Safety margin before 90s limit

    # 1. Parse Inputs
    if isinstance(distribution_spec, dict):
        if 'probabilities' in distribution_spec:
            probs = np.array(distribution_spec['probabilities'], dtype=float)
        elif 'p' in distribution_spec:
            probs = np.array(distribution_spec['p'], dtype=float)
        else:
            probs = np.array([distribution_spec.get(i, 0.0) for i in range(16)], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    else:
        probs = np.ones(16) / 16.0

    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = np.ones(16) / 16.0
    probs = np.clip(probs, 0, 1)

    if isinstance(support, (list, tuple, np.ndarray)):
        support_list = list(support)
    else:
        support_list = [(i // 4, i % 4) for i in range(16)]

    if len(support_list) != 16:
        support_list = [(i // 4, i % 4) for i in range(16)]

    V = np.zeros((16, 2))
    for i, (v1, v2) in enumerate(support_list):
        V[i, 0] = v1
        V[i, 1] = v2

    best_allocations = np.zeros((16, 2))
    best_payments = np.zeros(16)
    best_revenue = 0.0

    # Check if support is the standard grid
    is_grid = True
    for i, (v1, v2) in enumerate(support_list):
        if v1 != i // 4 or v2 != i % 4:
            is_grid = False
            break

    if is_grid:
        # --- Approach 1: Efficient 32-var LP using Monotonicity ---
        try:
            num_vars = 32
            # Indices:
            # x1_{i,j} -> 4*i + j
            # x2_{i,j} -> 16 + 4*i + j
            
            # Objective coefficients
            # Maximize sum_{i,j} pi_{i,j} * p_{i,j}
            # p_{i,j} = i*x1_{i,j} + j*x2_{i,j} - sum_{k=1}^i x1_{k-1,j} - sum_{k=1}^j x2_{i,k-1}
            # Coeff of x1_{i,j} in obj:
            # + i * pi_{i,j} (from p_{i,j})
            # - pi_{i',j} for all i' > i (from subtraction in p_{i',j})
            
            obj_c = np.zeros(num_vars)
            for i in range(4):
                for j in range(4):
                    idx_x1 = 4*i + j
                    idx_x2 = 16 + 4*i + j
                    
                    coeff_x1 = i * probs[i*4+j]
                    for ip in range(i+1, 4):
                        coeff_x1 -= probs[ip*4+j]
                    
                    coeff_x2 = j * probs[i*4+j]
                    for jp in range(j+1, 4):
                        coeff_x2 -= probs[i*4+jp]
                    
                    obj_c[idx_x1] = -coeff_x1  # linprog minimizes
                    obj_c[idx_x2] = -coeff_x2

            A_ub = []
            b_ub = []

            # Monotonicity Constraints
            # x1_{i,j} >= x1_{i-1,j}  => x1_{i-1,j} - x1_{i,j} <= 0
            # x1_{i,j} >= x1_{i,j-1}  => x1_{i,j-1} - x1_{i,j} <= 0
            # Same for x2
            
            for i in range(1, 4):
                for j in range(4):
                    row = np.zeros(num_vars)
                    row[4*(i-1) + j] = 1
                    row[4*i + j] = -1
                    A_ub.append(row)
                    b_ub.append(0)
                    
                    row2 = np.zeros(num_vars)
                    row2[16 + 4*(i-1) + j] = 1
                    row2[16 + 4*i + j] = -1
                    A_ub.append(row2)
                    b_ub.append(0)

            for i in range(4):
                for j in range(1, 4):
                    row = np.zeros(num_vars)
                    row[4*i + (j-1)] = 1
                    row[4*i + j] = -1
                    A_ub.append(row)
                    b_ub.append(0)
                    
                    row2 = np.zeros(num_vars)
                    row2[16 + 4*i + (j-1)] = 1
                    row2[16 + 4*i + j] = -1
                    A_ub.append(row2)
                    b_ub.append(0)

            # Non-negative Payment Constraints
            # p_{i,j} >= 0
            # i*x1_{i,j} + j*x2_{i,j} - sum_{k=1}^i x1_{k-1,j} - sum_{k=1}^j x2_{i,k-1} >= 0
            # => sum_{k=1}^i x1_{k-1,j} + sum_{k=1}^j x2_{i,k-1} - i*x1_{i,j} - j*x2_{i,j} <= 0
            
            for i in range(4):
                for j in range(4):
                    row = np.zeros(num_vars)
                    # -i * x1_{i,j}
                    row[4*i + j] = -i
                    # -j * x2_{i,j}
                    row[16 + 4*i + j] = -j
                    
                    # + sum_{k=1}^i x1_{k-1,j}
                    for k in range(1, i+1):
                        row[4*(k-1) + j] += 1
                        
                    # + sum_{k=1}^j x2_{i,k-1}
                    for k in range(1, j+1):
                        row[16 + 4*i + (k-1)] += 1
                        
                    A_ub.append(row)
                    b_ub.append(0)

            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            
            bounds = [(0, 1)] * num_vars

            result = linprog(obj_c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                             options={'presolve': True, 'dual_feasibility_tolerance': 1e-10,
                                      'primal_feasibility_tolerance': 1e-10,
                                      'dual_infeasibility_tolerance': 1e-10,
                                      'primal_infeasibility_tolerance': 1e-10,
                                      'ipm_optimality_tolerance': 1e-10})

            if result.success:
                x_opt = result.x
                x1_opt = np.clip(x_opt[0:16], 0, 1)
                x2_opt = np.clip(x_opt[16:32], 0, 1)

                allocations = np.zeros((16, 2))
                payments = np.zeros(16)

                for i in range(4):
                    for j in range(4):
                        idx = i*4 + j
                        allocations[idx, 0] = x1_opt[4*i + j]
                        allocations[idx, 1] = x2_opt[16 + 4*i + j]
                        
                        p_val = i * x1_opt[4*i + j] + j * x2_opt[16 + 4*i + j]
                        for k in range(1, i+1):
                            p_val -= x1_opt[4*(k-1) + j]
                        for k in range(1, j+1):
                            p_val -= x2_opt[16 + 4*i + (k-1)]
                        payments[idx] = max(0, p_val)

                # Verify
                valid = True
                for idx in range(16):
                    util = np.dot(V[idx], allocations[idx]) - payments[idx]
                    if util < -1e-8:
                        valid = False
                        break
                
                if valid:
                    for i in range(16):
                        for ip in range(16):
                            util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                            util_truth = np.dot(V[i], allocations[i]) - payments[i]
                            if util_report - util_truth > 1e-8:
                                valid = False
                                break
                        if not valid:
                            break

                if valid:
                    revenue = np.dot(probs, payments)
                    if revenue > best_revenue:
                        best_revenue = revenue
                        best_allocations = allocations.copy()
                        best_payments = payments.copy()

        except Exception:
            pass

    if time.time() > deadline:
        return {'allocations': best_allocations, 'payments': best_payments}

    # --- Approach 2: General IC LP (48 vars) Fallback ---
    # This is robust for non-grid supports or if the monotonicity approach fails
    try:
        n_vars = 48
        c = np.zeros(n_vars)
        for i in range(16):
            c[32 + i] = -probs[i]

        A_ir = np.zeros((16, n_vars))
        b_ir = np.zeros(16)
        for i in range(16):
            A_ir[i, 2*i] = -V[i, 0]
            A_ir[i, 2*i+1] = -V[i, 1]
            A_ir[i, 32+i] = 1

        n_ic = 16 * 16
        A_ic = np.zeros((n_ic, n_vars))
        b_ic = np.zeros(n_ic)

        for i in range(16):
            for ip in range(16):
                idx = i * 16 + ip
                A_ic[idx, 2*ip] += V[i, 0]
                A_ic[idx, 2*ip+1] += V[i, 1]
                A_ic[idx, 32+ip] += -1
                A_ic[idx, 2*i] += -V[i, 0]
                A_ic[idx, 2*i+1] += -V[i, 1]
                A_ic[idx, 32+i] += 1

        A = np.vstack([A_ir, A_ic])
        b = np.concatenate([b_ir, b_ic])

        bounds = []
        for i in range(16):
            bounds.append((0, 1))
            bounds.append((0, 1))
        for i in range(16):
            bounds.append((0, None))

        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                         options={'presolve': True, 'dual_feasibility_tolerance': 1e-10,
                                  'primal_feasibility_tolerance': 1e-10,
                                  'dual_infeasibility_tolerance': 1e-10,
                                  'primal_infeasibility_tolerance': 1e-10,
                                  'ipm_optimality_tolerance': 1e-10})

        if result.success:
            x = result.x
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            for i in range(16):
                allocations[i, 0] = x[2*i]
                allocations[i, 1] = x[2*i+1]
                payments[i] = x[32+i]

            allocations = np.clip(allocations, 0, 1)
            payments = np.maximum(payments, 0)

            valid = True
            for i in range(16):
                util = np.dot(V[i], allocations[i]) - payments[i]
                if util < -1e-8:
                    valid = False
                    break

            if valid:
                for i in range(16):
                    for ip in range(16):
                        util_report = np.dot(V[i], allocations[ip]) - payments[ip]
                        util_truth = np.dot(V[i], allocations[i]) - payments[i]
                        if util_report - util_truth > 1e-8:
                            valid = False
                            break
                    if not valid:
                        break

            if valid:
                revenue = np.dot(probs, payments)
                if revenue > best_revenue:
                    best_revenue = revenue
                    best_allocations = allocations.copy()
                    best_payments = payments.copy()

    except Exception:
        pass

    return {'allocations': best_allocations, 'payments': best_payments}
