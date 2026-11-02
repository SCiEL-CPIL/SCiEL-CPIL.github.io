import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    # Build Tensor T
    # T[a, b, c] = 1 if i_a==i_c, j_a==j_b, k_b==k_c
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    start_time = time.time()
    deadline = start_time + 145  # 5s buffer for return

    def check_decomposition(U, V, W, R):
        # U: (nm, R), V: (mp, R), W: (pn, R)
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((Rec - T)**2)
        if err > 1e-8:
            return False, err
        # Check if entries are close to half-integers
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        if np.allclose(U2, np.round(U2), atol=1e-5) and \
           np.allclose(V2, np.round(V2), atol=1e-5) and \
           np.allclose(W2, np.round(W2), atol=1e-5):
            # Snap to exact half-integers
            U_out = np.round(U2) / 2.0
            V_out = np.round(V2) / 2.0
            W_out = np.round(W2) / 2.0
            # Convert to list of lists
            return True, err, U_out, V_out, W_out
        return False, err, None, None, None

    def snap_and_check(U_flat, V_flat, W_flat, R):
        U = (np.round(U_flat * 2) / 2).reshape(n*m, R)
        V = (np.round(V_flat * 2) / 2).reshape(m*p, R)
        W = (np.round(W_flat * 2) / 2).reshape(p*n, R)
        valid, err, U_out, V_out, W_out = check_decomposition(U, V, W, R)
        return valid, U_out, V_out, W_out

    # Objective function: minimize squared error
    def objective(params, R):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((Rec - T)**2)
        return err

    def objective_with_penalty(params, R, lam):
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        U_flat = params[:size_U]
        V_flat = params[size_U:size_U+size_V]
        W_flat = params[size_U+size_V:]
        
        U = U_flat.reshape(n*m, R)
        V = V_flat.reshape(m*p, R)
        W = W_flat.reshape(p*n, R)
        
        Rec = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.sum((Rec - T)**2)
        
        # Penalty for non-half-integers: sin^2(pi * 2x) is 0 when 2x is integer
        # Using a smooth surrogate
        penalty_U = np.sum(np.sin(np.pi * 2 * U_flat)**2)
        penalty_V = np.sum(np.sin(np.pi * 2 * V_flat)**2)
        penalty_W = np.sum(np.sin(np.pi * 2 * W_flat)**2)
        
        return err + lam * (penalty_U + penalty_V + penalty_W)

    # We will try ranks from 29 up to 30 (schoolbook)
    # But 30 is guaranteed. 29 is the target.
    # Let's try to find a solution for rank 29.
    
    # Strategy: 
    # 1. Random restarts with L-BFGS-B on the raw error.
    # 2. If error is small, snap to half-integers and check.
    # 3. If that fails, use homotopy with penalty to guide towards half-integers.
    
    target_rank = 29
    size_U = n * m * target_rank
    size_V = m * p * target_rank
    size_W = p * n * target_rank
    total_vars = size_U + size_V + size_W

    # Pre-compute some random seeds
    seeds = [42 + i for i in range(50)]
    
    best_err = float('inf')
    best_params = None

    # Phase 1: Standard optimization minimizing error only
    for restart in range(len(seeds)):
        if time.time() > deadline - 10:
            break
        np.random.seed(seeds[restart])
        params = np.random.randn(total_vars) * 0.5
        
        try:
            res = minimize(objective, params, args=(target_rank,), method='L-BFGS-B', 
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.fun < best_err:
                best_err = res.fun
                best_params = res.x
                # Check if we found a solution
                if best_err < 1e-6:
                    valid, U_out, V_out, W_out = snap_and_check(
                        best_params[:size_U], best_params[size_U:size_U+size_V], 
                        best_params[size_U+size_V:], target_rank)
                    if valid:
                        return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
        except Exception:
            pass

    # If Phase 1 didn't find a valid half-integer solution, try Phase 2 with penalty
    # This helps if the solution is close to half-integers but not exactly, or if the optimizer got stuck in a local min with non-integer values.
    
    # Use the best params from Phase 1 as initial guess for Phase 2 if available, otherwise random
    if best_params is not None:
        init_params = best_params
    else:
        np.random.seed(123)
        init_params = np.random.randn(total_vars) * 0.5

    # Homotopy on lambda
    lambdas = [1.0, 10.0, 100.0, 1000.0]
    current_params = init_params.copy()
    
    for lam in lambdas:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(objective_with_penalty, current_params, args=(target_rank, lam), 
                           method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            current_params = res.x
            # Check after each step
            valid, U_out, V_out, W_out = snap_and_check(
                current_params[:size_U], current_params[size_U:size_U+size_V], 
                current_params[size_U+size_V:], target_rank)
            if valid:
                return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
        except Exception:
            pass

    # If we still haven't found rank 29, return schoolbook rank 30
    return {"U": U_sb, "V": V_sb, "W": W_sb}
