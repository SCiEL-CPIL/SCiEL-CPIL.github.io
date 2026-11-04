import time
import sys
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s limit to be safe
    
    n, m, p = DIMENSIONS
    
    # Helper to create schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return U, V, W
    
    # Verify a decomposition
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        # Build tensor T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
        # Should equal 1 for all (i,j,k)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Try to find a rank-r decomposition using numerical optimization
    def try_rank(r, max_iters=200, seed=42):
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Objective: minimize sum over (i,j,k) of (sum_r U_{i*m+j,r} V_{j*p+k,r} W_{k*n+i,r} - 1)^2
        def objective(params):
            # Unpack
            U_mat = params[:n*m*r].reshape(n*m, r)
            V_mat = params[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
            W_mat = params[n*m*r + m*p*r :].reshape(p*n, r)
            
            # Compute tensor entries
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for rr in range(r):
                            s += U_mat[i*m+j, rr] * V_mat[j*p+k, rr] * W_mat[k*n+i, rr]
                        err += (s - 1) ** 2
            return err
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Use Nelder-Mead for small r, L-BFGS-B for larger
        if r <= 30:
            result = minimize(objective, params, method='Nelder-Mead', 
                             options={'maxiter': max_iters, 'xatol': 1e-12, 'fatol': 1e-12})
        else:
            result = minimize(objective, params, method='L-BFGS-B', 
                             options={'maxiter': max_iters, 'gtol': 1e-12})
        
        if not result.success and result.fun > 1e-6:
            return None
        
        # Unpack
        U_mat = result.x[:n*m*r].reshape(n*m, r)
        V_mat = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W_mat = result.x[n*m*r + m*p*r :].reshape(p*n, r)
        
        # Snap to nearest integer/half-integer
        def snap_matrix(mat):
            mat_snapped = np.zeros_like(mat)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    val = mat[i, j]
                    # Try integer
                    nearest_int = round(val)
                    nearest_half = round(val * 2) / 2
                    if abs(val - nearest_int) < 1e-4:
                        mat_snapped[i, j] = nearest_int
                    elif abs(val - nearest_half) < 1e-4:
                        mat_snapped[i, j] = nearest_half
                    else:
                        # Keep as is if close enough
                        mat_snapped[i, j] = round(val * 2) / 2
            return mat_snapped
        
        U_snapped = snap_matrix(U_mat)
        V_snapped = snap_matrix(V_mat)
        W_snapped = snap_matrix(W_mat)
        
        # Verify
        if verify(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p):
            return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
        
        # If snapping failed, try the raw values
        if verify(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), n, m, p):
            return U_mat.tolist(), V_mat.tolist(), W_mat.tolist()
        
        return None
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = schoolbook(n, m, p)
    best_result = (U_sb, V_sb, W_sb)
    best_rank = n * m * p
    
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # Try ranks from 21 upward
    for rank in range(21, best_rank + 1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random seeds
        for seed in range(3):
            if time.time() > deadline - 10:
                break
            
            result = try_rank(rank, max_iters=150, seed=seed)
            if result is not None:
                U_found, V_found, W_found = result
                if verify(U_found, V_found, W_found, n, m, p):
                    best_result = (U_found, V_found, W_found)
                    best_rank = rank
                    break
        
        if best_rank == rank:
            break
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
