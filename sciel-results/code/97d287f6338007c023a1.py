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
    # Reshape for easier indexing
    U_mat = U.reshape(n, m, R)
    V_mat = V.reshape(m, p, R)
    W_mat = W.reshape(p, n, R)
    
    # Compute the tensor T[i, j, k] = sum_r W[k, i, r] * U[i, j, r] * V[j, k, r]
    # Note: W index in prompt is k*n+i. So W_mat[k, i, r] corresponds to W[k*n+i, r]
    # U index is i*m+j. U_mat[i, j, r] corresponds to U[i*m+j, r]
    # V index is j*p+k. V_mat[j, k, r] corresponds to V[j*p+k, r]
    
    # We need T[i, j, k] to be 1 for all i, j, k.
    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    # Use einsum for efficiency
    # U: (n, m, R)
    # V: (m, p, R)
    # W: (p, n, R)
    # Result T: (n, m, p)
    # T[i, j, k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    # einsum notation:
    # U: i j r
    # V: j k r
    # W: k i r
    # Output: i j k
    T = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat)
    
    # Check if T is all ones
    error = np.linalg.norm(T - 1.0)
    return error

def _snap_to_half(arr):
    return np.round(arr * 2) / 2

def _try_rank(R, n, m, p, time_limit=10):
    # Initialize
    # Start with random or scaled schoolbook?
    # Random is better for escaping local minima, but schoolbook might be a good basin.
    # Let's try a few random restarts.
    
    best_error = float('inf')
    best_factors = None
    
    start_time = time.time()
    num_restarts = 5
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        # Random initialization
        # Scale to 1
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Flatten
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
            res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
            if res.success:
                x_opt = res.x
                U_flat, V_flat, W_flat = x_opt[:n*m*R], x_opt[n*m*R:n*m*R+m*p*R], x_opt[n*m*R+m*p*R:]
                U_opt = U_flat.reshape(n*m, R)
                V_opt = V_flat.reshape(m*p, R)
                W_opt = W_flat.reshape(p*n, R)
                
                err = _verify(U_opt, V_opt, W_opt, n, m, p)
                if err < best_error:
                    best_error = err
                    best_factors = (U_opt, V_opt, W_opt)
                    
                    # Try snapping
                    U_snap = _snap_to_half(U_opt)
                    V_snap = _snap_to_half(V_opt)
                    W_snap = _snap_to_half(W_opt)
                    err_snap = _verify(U_snap, V_snap, W_snap, n, m, p)
                    if err_snap < 1e-6:
                        return U_snap, V_snap, W_snap
        except Exception:
            pass
            
    # If we found a good solution but didn't snap perfectly, try to snap the best one
    if best_factors is not None and best_error < 1e-4:
        U_opt, V_opt, W_opt = best_factors
        U_snap = _snap_to_half(U_opt)
        V_snap = _snap_to_half(V_opt)
        W_snap = _snap_to_half(W_opt)
        err_snap = _verify(U_snap, V_snap, W_snap, n, m, p)
        if err_snap < 1e-6:
            return U_snap, V_snap, W_snap
            
    return None

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook
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
                
    # Try to find lower rank
    # Known rank for 2x3x3 is likely 12.
    for R in range(12, rank_school + 1):
        try:
            res = _try_rank(R, n, m, p, time_limit=15)
            if res is not None:
                U, V, W = res
                # Convert to lists
                U_list = U.tolist()
                V_list = V.tolist()
                W_list = W.tolist()
                return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            continue
            
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
