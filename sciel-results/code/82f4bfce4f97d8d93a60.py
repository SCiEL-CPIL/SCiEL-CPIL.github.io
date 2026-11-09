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

def _try_rank(R, n, m, p, time_limit=10, num_restarts=5):
    best_error = float('inf')
    best_factors = None
    
    start_time = time.time()
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
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
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
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
    deadline = global_start + 130 # Safety margin
    
    # 1. Try Rank 15
    if time.time() < deadline - 40:
        res = _try_rank(15, n, m, p, time_limit=20, num_restarts=10)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 2. Try Rank 16
    if time.time() < deadline - 30:
        res = _try_rank(16, n, m, p, time_limit=15, num_restarts=10)
        if res is not None:
            U, V, W = res
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # 3. Try Rank 17
    if time.time() < deadline - 10:
        res = _try_rank(17, n, m, p, time_limit=10, num_restarts=10)
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
