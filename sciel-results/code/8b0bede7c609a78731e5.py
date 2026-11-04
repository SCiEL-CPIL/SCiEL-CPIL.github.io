import numpy as np
import time
import random
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 130.0  # 20 seconds margin from 150s limit
    
    def residuals(x, r):
        U = x[:r*n*m].reshape(r, n, m)
        V = x[r*n*m : r*n*m + r*m*p].reshape(r, m, p)
        W = x[r*n*m + r*m*p :].reshape(r, p, n)
        T = np.einsum('cij,cjk,cki->ijk', U, V, W)
        return (T - 1).flatten()
    
    def try_rank(r, max_restarts=50):
        best_solution = None
        best_residual = float('inf')
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(1000 + restart * 7)
            
            # Try different initialization strategies
            strategy = restart % 3
            
            if strategy == 0:
                # Small random
                U_flat = np.random.uniform(-0.5, 0.5, r * n * m)
                V_flat = np.random.uniform(-0.5, 0.5, r * m * p)
                W_flat = np.random.uniform(-0.5, 0.5, r * p * n)
            elif strategy == 1:
                # Sparse random
                U_flat = np.zeros(r * n * m)
                V_flat = np.zeros(r * m * p)
                W_flat = np.zeros(r * p * n)
                for idx in np.random.choice(r * n * m, size=r*n*m//2, replace=False):
                    U_flat[idx] = np.random.choice([-1, 1])
                for idx in np.random.choice(r * m * p, size=r*m*p//2, replace=False):
                    V_flat[idx] = np.random.choice([-1, 1])
                for idx in np.random.choice(r * p * n, size=r*p*n//2, replace=False):
                    W_flat[idx] = np.random.choice([-1, 1])
            else:
                # Structured: try to mimic Strassen-like patterns
                U_flat = np.random.uniform(-1, 1, r * n * m)
                V_flat = np.random.uniform(-1, 1, r * m * p)
                W_flat = np.random.uniform(-1, 1, r * p * n)
                # Add some structure
                for c in range(min(r, 7)):
                    # Set some entries to ±1
                    U_flat[c*n*m : (c+1)*n*m][np.random.choice(n*m, size=n*m//2, replace=False)] = 1.0
                    V_flat[c*m*p : (c+1)*m*p][np.random.choice(m*p, size=m*p//2, replace=False)] = 1.0
                    W_flat[c*p*n : (c+1)*p*n][np.random.choice(p*n, size=p*n//2, replace=False)] = 1.0
            
            x0 = np.concatenate([U_flat, V_flat, W_flat])
            
            try:
                res0 = residuals(x0, r)
                norm0 = np.linalg.norm(res0)
                
                if norm0 < best_residual:
                    best_residual = norm0
                    best_solution = x0.copy()
                
                if norm0 < 1e-10:
                    return x0, norm0
                
                # Use least_squares
                result = least_squares(
                    residuals, x0, args=(r,),
                    method='lm',
                    max_nfev=1000,
                    xtol=1e-12,
                    ftol=1e-12
                )
                
                if result.success or result.cost < best_residual:
                    norm_res = np.linalg.norm(result.fun)
                    if norm_res < best_residual:
                        best_residual = norm_res
                        best_solution = result.x.copy()
                    
                    if norm_res < 1e-10:
                        return result.x, norm_res
                        
            except Exception:
                continue
        
        return best_solution, best_residual
    
    # Try ranks from low to high
    for r in range(20, 37):
        if time.time() > deadline:
            break
        
        sol, res = try_rank(r, max_restarts=30)
        
        if sol is not None and res < 1e-9:
            U = sol[:r*n*m].reshape(r, n, m)
            V = sol[r*n*m : r*n*m + r*m*p].reshape(r, m, p)
            W = sol[r*n*m + r*m*p :].reshape(r, p, n)
            
            # Round to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            T_rounded = np.einsum('cij,cjk,cki->ijk', U_rounded, V_rounded, W_rounded)
            if np.allclose(T_rounded, 1, atol=1e-9):
                U_list = [[0]*r for _ in range(n*m)]
                V_list = [[0]*r for _ in range(m*p)]
                W_list = [[0]*r for _ in range(p*n)]
                
                for i in range(n):
                    for j in range(m):
                        for c in range(r):
                            U_list[i*m+j][c] = float(U_rounded[c, i, j])
                for j in range(m):
                    for k in range(p):
                        for c in range(r):
                            V_list[j*p+k][c] = float(V_rounded[c, j, k])
                for k in range(p):
                    for i in range(n):
                        for c in range(r):
                            W_list[k*n+i][c] = float(W_rounded[c, k, i])
                
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
    return {"U": U, "V": V, "W": W}
