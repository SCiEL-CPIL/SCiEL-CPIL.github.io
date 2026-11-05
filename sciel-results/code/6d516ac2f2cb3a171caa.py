import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    
    # Schoolbook rank
    schoolbook_rank = n * m * p  # 45
    
    # Build schoolbook decomposition
    def build_schoolbook():
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-9):
        U_reshaped = U.reshape(n, m, -1)
        V_reshaped = V.reshape(m, p, -1)
        W_reshaped = W.reshape(p, n, -1)
        
        # einsum indices: i j r, j k r, k i r -> i j k
        rec = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        
        return np.allclose(rec, 1.0, atol=tol)

    # Get the schoolbook solution as the best fallback
    best_U, best_V, best_W = build_schoolbook()
    
    # Objective function for a sub-problem
    # n, m, p are the dimensions of the sub-problem
    def objective_and_grad_sub(x, R, n_sub, m_sub, p_sub):
        U_flat = x[:n_sub*m_sub*R].reshape(n_sub*m_sub, R)
        V_flat = x[n_sub*m_sub*R:n_sub*m_sub*R + m_sub*p_sub*R].reshape(m_sub*p_sub, R)
        W_flat = x[n_sub*m_sub*R + m_sub*p_sub*R:].reshape(p_sub*n_sub, R)
        
        U_reshaped = U_flat.reshape(n_sub, m_sub, R)
        V_reshaped = V_flat.reshape(m_sub, p_sub, R)
        W_reshaped = W_flat.reshape(p_sub, n_sub, R)
        
        rec = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped, optimize=True)
        
        error_vec = rec - 1.0
        error = np.sum(error_vec**2)
        
        grad_U = 2.0 * np.einsum('ijk,jkr,kir->ijr', error_vec, V_reshaped, W_reshaped, optimize=True)
        grad_V = 2.0 * np.einsum('ijk,ijr,kir->jkr', error_vec, U_reshaped, W_reshaped, optimize=True)
        grad_W = 2.0 * np.einsum('ijk,ijr,jkr->kir', error_vec, U_reshaped, V_reshaped, optimize=True)
        
        return error, np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])

    # Helper to solve a sub-problem
    def solve_subproblem(n_sub, m_sub, p_sub, target_rank, max_iter=200):
        # Initialize with schoolbook
        R_init = n_sub * m_sub * p_sub
        U_init = np.zeros((n_sub*m_sub, R_init))
        V_init = np.zeros((m_sub*p_sub, R_init))
        W_init = np.zeros((p_sub*n_sub, R_init))
        col = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U_init[i*m_sub+j, col] = 1
                    V_init[j*p_sub+k, col] = 1
                    W_init[k*n_sub+i, col] = 1
                    col += 1
        
        # Try to reduce rank
        # We start from schoolbook and try to zero out some columns?
        # Or just optimize for the target rank directly with random init?
        # Random init is often better for finding low rank solutions.
        
        # Strategy: Try to find a decomposition of rank `target_rank`
        # We use L-BFGS-B
        
        best_err = np.inf
        best_x = None
        
        # Try a few random restarts
        for restart in range(5):
            if time.time() > deadline - 10:
                break
                
            # Random initialization
            U_rand = np.random.randn(n_sub*m_sub, target_rank)
            V_rand = np.random.randn(m_sub*p_sub, target_rank)
            W_rand = np.random.randn(p_sub*n_sub, target_rank)
            
            x0 = np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()])
            
            try:
                from scipy.optimize import minimize
                
                def obj(x):
                    e, _ = objective_and_grad_sub(x, target_rank, n_sub, m_sub, p_sub)
                    return e
                
                def jac(x):
                    _, g = objective_and_grad_sub(x, target_rank, n_sub, m_sub, p_sub)
                    return g
                
                res = minimize(obj, x0, jac=jac, method='L-BFGS-B', 
                               options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8})
                
                if res.fun < best_err:
                    best_err = res.fun
                    best_x = res.x.copy()
                    
                # Check if we found a solution
                if res.fun < 1e-8:
                    U_sol = res.x[:n_sub*m_sub*target_rank].reshape(n_sub*m_sub, target_rank)
                    V_sol = res.x[n_sub*m_sub*target_rank:n_sub*m_sub*target_rank + m_sub*p_sub*target_rank].reshape(m_sub*p_sub, target_rank)
                    W_sol = res.x[n_sub*m_sub*target_rank + m_sub*p_sub*target_rank:].reshape(p_sub*n_sub, target_rank)
                    
                    # Round to half-integers
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    # Verify
                    U_r = U_rounded.reshape(n_sub, m_sub, target_rank)
                    V_r = V_rounded.reshape(m_sub, p_sub, target_rank)
                    W_r = W_rounded.reshape(p_sub, n_sub, target_rank)
                    rec = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r, optimize=True)
                    
                    if np.allclose(rec, 1.0, atol=1e-9):
                        return U_rounded, V_rounded, W_rounded
                        
            except Exception:
                pass
        
        return None

    # Solve 3x3x3 sub-problem
    U1 = V1 = W1 = None
    for R1 in range(20, 28): # Try 20 to 27
        if time.time() > deadline - 20:
            break
        sol1 = solve_subproblem(3, 3, 3, R1, max_iter=100)
        if sol1 is not None:
            U1, V1, W1 = sol1
            break
            
    # Solve 3x3x2 sub-problem
    U2 = V2 = W2 = None
    for R2 in range(10, 19): # Try 10 to 18
        if time.time() > deadline - 10:
            break
        sol2 = solve_subproblem(3, 3, 2, R2, max_iter=100)
        if sol2 is not None:
            U2, V2, W2 = sol2
            break
            
    # Construct full solution
    # If we found solutions for both, concatenate
    if U1 is not None and U2 is not None:
        # U1 is 9xR1, U2 is 9xR2
        U_full = np.hstack([U1, U2])
        
        # V1 is 9xR1 (for 3x3x3, B is 3x3). V2 is 6xR2 (for 3x3x2, B is 3x2).
        # Full V is 15x(R1+R2)
        V_full = np.zeros((15, U_full.shape[1]))
        V_full[:9, :U1.shape[1]] = V1
        V_full[9:, U1.shape[1]:] = V2
        
        # W1 is 9xR1. W2 is 6xR2.
        # Full W is 15x(R1+R2)
        W_full = np.zeros((15, U_full.shape[1]))
        W_full[:9, :U1.shape[1]] = W1
        W_full[9:, U1.shape[1]:] = W2
        
        if verify(U_full, V_full, W_full):
            return {
                "U": U_full.tolist(),
                "V": V_full.tolist(),
                "W": W_full.tolist()
            }
            
    # Fallback to schoolbook
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
