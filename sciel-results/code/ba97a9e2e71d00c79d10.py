import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    T_LIMIT = 85.0
    types = [(a, b) for a in range(4) for b in range(4)]
    N = 16
    V = np.array([[types[i][0], types[i][1]] for i in range(N)], dtype=np.float64)

    def _elapsed():
        return time.time() - t0

    def _parse_probs():
        if isinstance(distribution_spec, dict):
            if 'probs' in distribution_spec:
                arr = np.asarray(distribution_spec['probs'], dtype=np.float64)
                if arr.shape == (N,):
                    arr = np.maximum(arr, 0.0)
                    s = arr.sum()
                    return arr / s if s > 0 else np.ones(N) / N
            if 'distribution' in distribution_spec:
                d = distribution_spec['distribution']
                if d == 'uniform':
                    return np.ones(N) / N
                if d == 'geometric':
                    base = 0.7
                    arr = np.array([base**types[i][0] * base**types[i][1] for i in range(N)])
                    return arr / arr.sum()
        if isinstance(distribution_spec, (list, tuple, np.ndarray)):
            arr = np.asarray(distribution_spec, dtype=np.float64)
            if arr.shape == (N,):
                arr = np.maximum(arr, 0.0)
                s = arr.sum()
                if s > 0:
                    return arr / s
        return np.ones(N) / N

    probs = _parse_probs()
    if _elapsed() > T_LIMIT:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    # First, solve the allocation LP (x, y) with IC and IR constraints
    # Variables: [x_0..x_15, y_0..y_15]
    # Objective: Max sum_i probs[i] * (x[i]*v1_i + y[i]*v2_i) - expected payment
    # But we need to think about this differently.
    
    # Let's use the dual approach: the optimal payment for type i given allocation a_i
    # is determined by the IC constraints. For a monotone allocation, the payment is:
    # p_i = v_i . a_i - integral of (a_j - a_i) . dv_j along path from reference type
    
    # For now, let's solve the full LP with tighter tolerances and add a refinement pass
    
    # LP variables: [x_0..x_15, y_0..y_15, p_0..p_15]
    # Revenue = sum_i probs[i] * p[i], so minimize -sum_i probs[i]*p[i]
    c_obj = np.zeros(3 * N)
    c_obj[2 * N:3 * N] = -probs  # minimize negative revenue

    # IR: for each i, p[i] <= x[i]*v1_i + y[i]*v2_i
    # => -v1_i*x[i] - v2_i*y[i] + p[i] <= 0
    A_ir = np.zeros((N, 3 * N))
    for i in range(N):
        A_ir[i, i] = -V[i, 0]
        A_ir[i, N + i] = -V[i, 1]
        A_ir[i, 2 * N + i] = 1.0
    b_ir = np.zeros(N)

    # IC: for each (i,j), i!=j:
    # (x[j]-x[i])*v1_i + (y[j]-y[i])*v2_i - p[j] + p[i] <= 0
    # Build as matrix
    n_ic = N * (N - 1)
    A_ic = np.zeros((n_ic, 3 * N))
    idx = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            A_ic[idx, j] = V[i, 0]
            A_ic[idx, i] = -V[i, 0]
            A_ic[idx, N + j] = V[i, 1]
            A_ic[idx, N + i] = -V[i, 1]
            A_ic[idx, 2 * N + j] = -1.0
            A_ic[idx, 2 * N + i] = 1.0
            idx += 1
    b_ic = np.zeros(n_ic)

    A_ub = np.vstack([A_ir, A_ic])
    b_ub = np.concatenate([b_ir, b_ic])

    # Payments can be at most the value of the allocation for IR
    max_payment = np.array([V[i, 0] + V[i, 1] for i in range(N)])
    bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, max_payment[i]) for i in range(N)]

    if _elapsed() > T_LIMIT:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-14,
                               'primal_feasibility_tolerance': 1e-14})
    except Exception:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    if not res.success or res.x is None:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    x = np.clip(res.x[:N], 0.0, 1.0)
    y = np.clip(res.x[N:2 * N], 0.0, 1.0)
    p = np.maximum(res.x[2 * N:3 * N], 0.0)

    # Post-process: enforce IR by capping payments
    v1 = V[:, 0]
    v2 = V[:, 1]
    p_cap = x * v1 + y * v2
    p = np.minimum(p, p_cap)
    p = np.maximum(p, 0.0)

    # Refine Payments with Secondary LP
    # Fix x, y. Optimize p.
    # Objective: Max sum(probs * p) -> Min -sum(probs * p)
    c_p = -probs
    
    # Constraints for p:
    # IR: p_i <= V_i . a_i  =>  p_i - V_i . a_i <= 0
    A_p_ir = np.zeros((N, N))
    b_p_ir = np.zeros(N)
    for i in range(N):
        A_p_ir[i, i] = 1.0
        b_p_ir[i] = x[i] * V[i, 0] + y[i] * V[i, 1]
        
    # IC: p_i - p_j <= V_i . (a_i - a_j)
    # => p_i - p_j - V_i . (a_i - a_j) <= 0
    A_p_ic = np.zeros((n_ic, N))
    b_p_ic = np.zeros(n_ic)
    idx = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            A_p_ic[idx, i] = 1.0
            A_p_ic[idx, j] = -1.0
            # RHS: V_i . (a_i - a_j)
            val_i = x[i] * V[i, 0] + y[i] * V[i, 1]
            val_j = x[j] * V[i, 0] + y[j] * V[i, 1]
            b_p_ic[idx] = val_i - val_j
            idx += 1
            
    A_ub_p = np.vstack([A_p_ir, A_p_ic])
    b_ub_p = np.concatenate([b_p_ir, b_p_ic])
    
    bounds_p = [(0.0, b_p_ir[i]) for i in range(N)]
    
    if _elapsed() < T_LIMIT - 5.0:
        try:
            res_p = linprog(c_p, A_ub=A_ub_p, b_ub=b_ub_p, bounds=bounds_p, method='highs',
                            options={'presolve': True, 'dual_feasibility_tolerance': 1e-14,
                                     'primal_feasibility_tolerance': 1e-14})
            if res_p.success and res_p.x is not None:
                p_refined = np.maximum(res_p.x, 0.0)
                # Ensure IR is strictly satisfied (numerical safety)
                p_cap = x * V[:, 0] + y * V[:, 1]
                p_refined = np.minimum(p_refined, p_cap)
                p = p_refined
        except Exception:
            pass # Keep original p

    # New: Try to improve allocations using a local search on the allocation variables
    # while keeping payments optimized
    if _elapsed() < T_LIMIT - 10.0:
        # Try perturbing allocations slightly and re-solving for payments
        best_p = p.copy()
        best_rev = np.dot(probs, p)
        
        # Try small perturbations in allocation space
        np.random.seed(42)
        for trial in range(20):
            if _elapsed() > T_LIMIT - 5.0:
                break
                
            # Create a perturbed allocation
            x_trial = x.copy()
            y_trial = y.copy()
            
            # Random small perturbation
            mask = np.random.random(N) > 0.8  # Perturb ~20% of types
            x_trial[mask] = np.clip(x_trial[mask] + np.random.uniform(-0.1, 0.1, mask.sum()), 0.0, 1.0)
            y_trial[mask] = np.clip(y_trial[mask] + np.random.uniform(-0.1, 0.1, mask.sum()), 0.0, 1.0)
            
            # Solve for optimal payments given these allocations
            c_p_trial = -probs
            A_p_ir_trial = np.zeros((N, N))
            b_p_ir_trial = np.zeros(N)
            for i in range(N):
                A_p_ir_trial[i, i] = 1.0
                b_p_ir_trial[i] = x_trial[i] * V[i, 0] + y_trial[i] * V[i, 1]
                
            A_p_ic_trial = np.zeros((n_ic, N))
            b_p_ic_trial = np.zeros(n_ic)
            idx = 0
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    A_p_ic_trial[idx, i] = 1.0
                    A_p_ic_trial[idx, j] = -1.0
                    val_i = x_trial[i] * V[i, 0] + y_trial[i] * V[i, 1]
                    val_j = x_trial[j] * V[i, 0] + y_trial[j] * V[i, 1]
                    b_p_ic_trial[idx] = val_i - val_j
                    idx += 1
                    
            A_ub_p_trial = np.vstack([A_p_ir_trial, A_p_ic_trial])
            b_ub_p_trial = np.concatenate([b_p_ir_trial, b_p_ic_trial])
            bounds_p_trial = [(0.0, b_p_ir_trial[i]) for i in range(N)]
            
            try:
                res_p_trial = linprog(c_p_trial, A_ub=A_ub_p_trial, b_ub=b_ub_p_trial, 
                                      bounds=bounds_p_trial, method='highs',
                                      options={'presolve': True, 'dual_feasibility_tolerance': 1e-14,
                                               'primal_feasibility_tolerance': 1e-14})
                if res_p_trial.success and res_p_trial.x is not None:
                    p_trial = np.maximum(res_p_trial.x, 0.0)
                    p_trial = np.minimum(p_trial, x_trial * V[:, 0] + y_trial * V[:, 1])
                    p_trial = np.maximum(p_trial, 0.0)
                    
                    # Verify IC
                    ic_ok = True
                    for i in range(N):
                        if x_trial[i] * V[i, 0] + y_trial[i] * V[i, 1] - p_trial[i] < -1e-8:
                            ic_ok = False
                            break
                        for j in range(N):
                            if i == j:
                                continue
                            ur = x_trial[j] * V[i, 0] + y_trial[j] * V[i, 1] - p_trial[j]
                            ut = x_trial[i] * V[i, 0] + y_trial[i] * V[i, 1] - p_trial[i]
                            if ur - ut > 1e-8:
                                ic_ok = False
                                break
                        if not ic_ok:
                            break
                    
                    if ic_ok:
                        rev_trial = np.dot(probs, p_trial)
                        if rev_trial > best_rev + 1e-10:
                            best_rev = rev_trial
                            best_p = p_trial.copy()
                            x = x_trial.copy()
                            y = y_trial.copy()
            except Exception:
                pass
        
        p = best_p

    # Final verification
    ic_ok = True
    for i in range(N):
        if x[i] * V[i, 0] + y[i] * V[i, 1] - p[i] < -1e-8:
            ic_ok = False
            break
        for j in range(N):
            if i == j:
                continue
            ur = x[j] * V[i, 0] + y[j] * V[i, 1] - p[j]
            ut = x[i] * V[i, 0] + y[i] * V[i, 1] - p[i]
            if ur - ut > 1e-8:
                ic_ok = False
                break
        if not ic_ok:
            break

    if not ic_ok:
        return {'allocations': np.zeros((N, 2)), 'payments': np.zeros(N)}

    return {'allocations': np.column_stack((x, y)), 'payments': p}
