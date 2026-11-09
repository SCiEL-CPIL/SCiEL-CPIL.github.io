import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n, m, p):
    """
    Verify if the decomposition is exact.
    U: (n*m, R)
    V: (m*p, R)
    W: (p*n, R)
    Check: sum_r W[k*n+i, r] * U[i*m+j, r] * V[j*p+k, r] == 1 for all i,j,k
    Returns the error norm.
    """
    R = U.shape[1]
    U_mat = U.reshape(n, m, R)
    V_mat = V.reshape(m, p, R)
    W_mat = W.reshape(p, n, R)
    
    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
    error = np.linalg.norm(T - 1.0)
    return error

def _snap_to_half(arr):
    return np.round(arr * 2) / 2

def _local_search_snapping(U, V, W, n, m, p, max_iters=20, tol=1e-9):
    """
    Attempt to refine a snapped solution to exact half-integers via local search.
    Assumes U, V, W are already snapped to half-integers.
    """
    U_s = U.copy()
    V_s = V.copy()
    W_s = W.copy()
    
    best_err = _verify(U_s, V_s, W_s, n, m, p)
    
    if best_err < tol:
        return U_s, V_s, W_s, best_err

    for _ in range(max_iters):
        improved = False
        
        # Helper to try updating a single matrix
        def try_update(mat, name):
            nonlocal improved, best_err
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    orig_val = mat[i, j]
                    # Try neighbors in the half-integer grid: -0.5, 0, +0.5
                    # Since it's already snapped, we look at current +/- 0.5
                    candidates = [orig_val - 0.5, orig_val + 0.5]
                    for cand in candidates:
                        mat[i, j] = cand
                        err = _verify(U_s, V_s, W_s, n, m, p)
                        if err < best_err - 1e-12:
                            best_err = err
                            improved = True
                        else:
                            mat[i, j] = orig_val
                            
        try_update(U_s, 'U')
        try_update(V_s, 'V')
        try_update(W_s, 'W')
        
        if not improved:
            break
            
    return U_s, V_s, W_s, best_err

def _try_rank(R, n, m, p, time_limit=15, num_restarts=10):
    best_error = float('inf')
    best_factors = None
    
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialization strategies
        # 1. Random Gaussian
        # 2. Schoolbook-based perturbation (for lower ranks, this is tricky, but random is safer)
        # Let's use random but scaled appropriately
        scale = 1.0
        U = np.random.randn(n * m, R) * scale
        V = np.random.randn(m * p, R) * scale
        W = np.random.randn(p * n, R) * scale
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(x):
            U_flat, V_flat, W_flat = x[:n*m*R], x[n*m*R:n*m*R+m*p*R], x[n*m*R+m*p*R:]
            U_mat = U_flat.reshape(n, m, R)
            V_mat = V_flat.reshape(m, p, R)
            W_mat = W_flat.reshape(p, n, R)
            
            T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
            err = np.sum((T - 1.0)**2)
            return err
        
        try:
            # Use L-BFGS-B
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            
            x_opt = res.x
            U_flat, V_flat, W_flat = x_opt[:n*m*R], x_opt[n*m*R:n*m*R+m*p*R], x_opt[n*m*R+m*p*R:]
            U_opt = U_flat.reshape(n*m, R)
            V_opt = V_flat.reshape(m*p, R)
            W_opt = W_flat.reshape(p*n, R)
            
            # Snap and verify
            U_snap = _snap_to_half(U_opt)
            V_snap = _snap_to_half(V_opt)
            W_snap = _snap_to_half(W_opt)
            
            err_snap = _verify(U_snap, V_snap, W_snap, n, m, p)
            
            # If close, try to polish with local search
            if err_snap < 0.1:
                U_pol, V_pol, W_pol, err_pol = _local_search_snapping(U_snap, V_snap, W_snap, n, m, p)
                if err_pol < err_snap:
                    U_snap, V_snap, W_snap = U_pol, V_pol, W_pol
                    err_snap = err_pol
            
            if err_snap < 1e-9:
                return U_snap, V_snap, W_snap
                
            if err_snap < best_error:
                best_error = err_snap
                best_factors = (U_snap, V_snap, W_snap)
                
        except Exception:
            pass
            
    return None

def construct():
    n, m, p = DIMENSIONS
    global_start = time.time()
    deadline = global_start + 140 # Safety margin for 150s limit
    
    # Try ranks in ascending order
    # Theoretical lower bound is ~13.5, so 14 is the first candidate.
    # Rank 18 is schoolbook.
    ranks_to_try = [14, 15, 16, 17, 18]
    
    for R in ranks_to_try:
        if time.time() >= deadline - 5:
            break
            
        # Allocate time based on rank difficulty
        # Lower ranks are harder to find, give them more time if available
        remaining_time = deadline - time.time() - 5
        if R <= 15:
            t_limit = min(20, remaining_time * 0.4)
            n_restarts = 15
        else:
            t_limit = min(10, remaining_time * 0.2)
            n_restarts = 5
            
        if t_limit < 1:
            continue
            
        res = _try_rank(R, n, m, p, time_limit=t_limit, num_restarts=n_restarts)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
                
    return {"U": U_school, "V": V_school, "W": W_school}
