import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition
    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return U, V, W
    
    # Verify a decomposition
    def verify(U, V, W, n, m, p):
        rank = len(U[0])
        U_3d = np.array(U, dtype=float).reshape(n, m, rank)
        V_3d = np.array(V, dtype=float).reshape(m, p, rank)
        W_3d = np.array(W, dtype=float).reshape(p, n, rank)
        
        T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
        return np.allclose(T_recon, 1.0, atol=1e-8)
    
    # First, get the schoolbook solution as fallback
    U_school, V_school, W_school = get_schoolbook(n, m, p)
    best_rank = n * m * p  # 72
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Try to find a better decomposition using optimization
    # We will try ranks from 60 down to 23
    # For each rank, we will try a few random initializations
    
    def try_find_decomposition(target_rank, max_iter=200, tol=1e-8):
        if time.time() > deadline:
            return None
        
        # Flatten U, V, W into a single vector x
        # U: n*m*target_rank
        # V: m*p*target_rank
        # W: p*n*target_rank
        total_vars = (n*m + m*p + p*n) * target_rank
        
        def residuals(x):
            U_flat = x[:n*m*target_rank]
            V_flat = x[n*m*target_rank:n*m*target_rank + m*p*target_rank]
            W_flat = x[n*m*target_rank + m*p*target_rank:]
            
            U_3d = U_flat.reshape(n, m, target_rank)
            V_3d = V_flat.reshape(m, p, target_rank)
            W_3d = W_flat.reshape(p, n, target_rank)
            
            T_recon = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
            # We want T_recon to be 1.0 everywhere
            return T_recon.ravel() - 1.0
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline:
                break
            
            np.random.seed(42 + attempt)
            x0 = np.random.randn(total_vars) * 0.1
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=max_iter)
                if result.success or np.linalg.norm(result.fun) < tol:
                    U_flat = result.x[:n*m*target_rank]
                    V_flat = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank]
                    W_flat = result.x[n*m*target_rank + m*p*target_rank:]
                    
                    U = U_flat.reshape(n*m, target_rank)
                    V = V_flat.reshape(m*p, target_rank)
                    W = W_flat.reshape(p*n, target_rank)
                    
                    # Convert to lists of lists
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    
                    # Verify
                    if verify(U_list, V_list, W_list, n, m, p):
                        return U_list, V_list, W_list
            except Exception:
                pass
        
        return None
    
    # Try ranks from 60 down to 23
    for r in range(60, 22, -1):
        if time.time() > deadline:
            break
        
        result = try_find_decomposition(r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break  # Found a valid decomposition, stop
    
    return {"U": best_U, "V": best_V, "W": best_W}
