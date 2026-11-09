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
    """
    R = U.shape[1]
    U_mat = U.reshape(n, m, R)
    V_mat = V.reshape(m, p, R)
    W_mat = W.reshape(p, n, R)
    
    T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
    error = np.linalg.norm(T - 1.0)
    return error

def _snap_to_half(arr):
    return np.round(arr * 2) / 2

def _build_schoolbook(n, m, p, R):
    """Build a schoolbook-like initialization for rank R."""
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    
    # Use first n*m*p entries as schoolbook
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i*m*p + j*p + k
                if col < R:
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
    
    # For remaining columns, add small random perturbations to break symmetry
    for col in range(n*m*p, R):
        U[:, col] = np.random.randn(n*m) * 0.1
        V[:, col] = np.random.randn(m*p) * 0.1
        W[:, col] = np.random.randn(p*n) * 0.1
    
    return U, V, W

def _try_rank(R, n, m, p, time_limit=10, num_restarts=5):
    best_error = float('inf')
    best_factors = None
    
    start_time = time.time()
    
    def objective(x):
        U_flat, V_flat, W_flat = x[:n*m*R], x[n*m*R:n*m*R+m*p*R], x[n*m*R+m*p*R:]
        U_mat = U_flat.reshape(n, m, R)
        V_mat = V_flat.reshape(m, p, R)
        W_mat = W_flat.reshape(p, n, R)
        
        T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
        err = np.sum((T - 1.0)**2)
        return err
    
    # Strategy 1: Start from schoolbook initialization
    if time.time() - start_time < time_limit * 0.5:
        U0, V0, W0 = _build_schoolbook(n, m, p, R)
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            x_opt = res.x
            U_opt = x_opt[:n*m*R].reshape(n*m, R)
            V_opt = x_opt[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
            W_opt = x_opt[n*m*R+m*p*R:].reshape(p*n, R)
            
            # Try snapping to half-integers
            U_snap = _snap_to_half(U_opt)
            V_snap = _snap_to_half(V_opt)
            W_snap = _snap_to_half(W_opt)
            
            err_snap = _verify(U_snap, V_snap, W_snap, n, m, p)
            
            if err_snap < 1e-6:
                return U_snap, V_snap, W_snap
                
            if err_snap < best_error:
                best_error = err_snap
                best_factors = (U_snap, V_snap, W_snap)
                
            # Also try without snapping to see if we're close
            err_unsnap = _verify(U_opt, V_opt, W_opt, n, m, p)
            if err_unsnap < 1e-8:
                # Very close, try more aggressive snapping or polish
                for snap_factor in [1.0, 0.5, 0.25, 0.125]:
                    U_try = np.round(U_opt / snap_factor) * snap_factor
                    V_try = np.round(V_opt / snap_factor) * snap_factor
                    W_try = np.round(W_opt / snap_factor) * snap_factor
                    err_try = _verify(U_try, V_try, W_try, n, m, p)
                    if err_try < 1e-6:
                        return U_try, V_try, W_try
                    if err_try < best_error:
                        best_error = err_try
                        best_factors = (U_try, V_try, W_try)
                        
        except Exception:
            pass
    
    # Strategy 2: Random initializations
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            x_opt = res.x
            U_opt = x_opt[:n*m*R].reshape(n*m, R)
            V_opt = x_opt[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
            W_opt = x_opt[n*m*R+m*p*R:].reshape(p*n, R)
            
            U_snap = _snap_to_half(U_opt)
            V_snap = _snap_to_half(V_opt)
            W_snap = _snap_to_half(W_opt)
            
            err_snap = _verify(U_snap, V_snap, W_snap, n, m, p)
            
            if err_snap < 1e-6:
                return U_snap, V_snap, W_snap
                
            if err_snap < best_error:
                best_error = err_snap
                best_factors = (U_snap, V_snap, W_snap)
                
        except Exception:
            pass
            
    return best_factors if best_error < 1e-6 else None

def construct():
    n, m, p = DIMENSIONS
    global_start = time.time()
    deadline = global_start + 140  # Safety margin for 150s limit
    
    # Try ranks from low to high, with more time for lower ranks
    rank_configs = [
        (15, 40, 15),  # (rank, time_limit, num_restarts)
        (16, 30, 15),
        (17, 25, 12),
    ]
    
    for rank, time_limit, num_restarts in rank_configs:
        if time.time() > deadline - 5:
            break
            
        res = _try_rank(rank, n, m, p, time_limit=time_limit, num_restarts=num_restarts)
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
