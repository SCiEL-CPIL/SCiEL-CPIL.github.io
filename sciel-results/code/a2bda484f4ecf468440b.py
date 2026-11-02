import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Try to find a low-rank decomposition for 2x3x3 first
    # If found, embed it for 2x3x6
    # If not, fall back to schoolbook 2x3x6
    
    # Target: Rank 11 for 2x3x3
    n, m, p = 2, 3, 3
    R_target = 11
    
    # Build target tensor T for 2x3x3
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    best_solution = None
    best_rank = n*m*p # 36 for 2x3x6, but we are searching 2x3x3
    
    # Search for 2x3x3 decomposition
    for R in range(11, 19): # Try ranks 11 to 18
        if time.time() > deadline - 20:
            break
        
        found = False
        for restart in range(20):
            if time.time() > deadline - 20:
                break
            
            # Initialize
            U = np.random.randn(n*m, R)
            V = np.random.randn(m*p, R)
            W = np.random.randn(p*n, R)
            
            # Objective function
            def objective(params):
                U_flat = params[:n*m*R].reshape(n*m, R)
                V_flat = params[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_flat = params[n*m*R + m*p*R:].reshape(p*n, R)
                
                # Compute tensor sum
                # T_sum[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
                # Efficient computation:
                # T_sum = np.einsum('ir, jr, kr -> ijk', U_flat, V_flat, W_flat)
                T_sum = np.einsum('ir,jr,kr->ijk', U_flat, V_flat, W_flat)
                residual = T_sum - T
                return np.sum(residual**2)
            
            # Initial guess
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 50, 'ftol': 1e-10})
                if res.success or res.fun < 1e-6:
                    # Round to half-integers
                    U_opt = res.x[:n*m*R].reshape(n*m, R)
                    V_opt = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W_opt = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    U_rounded = np.round(U_opt * 2) / 2
                    V_rounded = np.round(V_opt * 2) / 2
                    W_rounded = np.round(W_opt * 2) / 2
                    
                    # Check exactness
                    T_check = np.einsum('ir,jr,kr->ijk', U_rounded, V_rounded, W_rounded)
                    if np.allclose(T_check, T, atol=1e-9):
                        best_solution = (U_rounded, V_rounded, W_rounded, R)
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            break
    
    if best_solution:
        U_233, V_233, W_233, R_233 = best_solution
        # Embed into 2x3x6
        # Rank is 2 * R_233
        R_236 = 2 * R_233
        n, m, p = 2, 3, 6
        
        U_236 = np.zeros((n*m, R_236))
        V_236 = np.zeros((m*p, R_236))
        W_236 = np.zeros((p*n, R_236))
        
        # First half
        U_236[:, :R_233] = U_233
        V_236[:m*3, :R_233] = V_233 # Top 9 rows of V (3*3)
        W_236[:3*n, :R_233] = W_233 # Top 6 rows of W (3*2)
        
        # Second half
        U_236[:, R_233:] = U_233
        V_236[m*3:, R_233:] = V_233 # Bottom 9 rows of V
        W_236[3*n:, R_233:] = W_233 # Bottom 6 rows of W
        
        return {
            "U": U_236.tolist(),
            "V": V_236.tolist(),
            "W": W_236.tolist()
        }
    else:
        # Fallback to schoolbook
        n, m, p = DIMENSIONS
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
