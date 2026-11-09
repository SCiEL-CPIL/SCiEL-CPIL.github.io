"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_school
    
    # Try lower ranks with multiple random restarts
    for target_rank in [110, 100, 90, 80]:
        if time.time() > deadline - 30:
            break
        if target_rank >= best_rank:
            continue
        
        success = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(42 + restart)
            scale = 1.0 / np.sqrt(target_rank)
            U_init = np.random.randn(n * m, target_rank) * scale
            V_init = np.random.randn(m * p, target_rank) * scale
            W_init = np.random.randn(p * n, target_rank) * scale
            
            def residuals(params):
                U_flat = params[:n*m*target_rank]
                V_flat = params[n*m*target_rank : n*m*target_rank + m*p*target_rank]
                W_flat = params[n*m*target_rank + m*p*target_rank :]
                
                U_mat = U_flat.reshape(n*m, target_rank)
                V_mat = V_flat.reshape(m*p, target_rank)
                W_mat = W_flat.reshape(p*n, target_rank)
                
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_mat[i*m+j, :] * V_mat[j*p+k, :] * W_mat[k*n+i, :])
                            res.append(val - 1)
                return res
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = least_squares(residuals, x0, method='lm', max_nfev=2000)
                if result.cost < 1e-6:
                    U_sol = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_sol = result.x[n*m*target_rank : n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
                    W_sol = result.x[n*m*target_rank + m*p*target_rank :].reshape(p*n, target_rank)
                    
                    best_U, best_V, best_W = U_sol, V_sol, W_sol
                    best_rank = target_rank
                    success = True
                    break
            except:
                pass
        
        if success:
            continue
        else:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
