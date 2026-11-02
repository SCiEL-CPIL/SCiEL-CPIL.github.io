import numpy as np
from scipy.optimize import fsolve
import time

DIMENSIONS = [2, 2, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    start_time = time.time()
    deadline = start_time + 120
    
    # Try to find lower rank decompositions
    for rank in range(25, rank_sb):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for trial in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Define the residual function
            def residuals(params):
                U_flat = params[:n*m*rank].reshape(n*m, rank)
                V_flat = params[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                W_flat = params[n*m*rank+m*p*rank:].reshape(p*n, rank)
                
                res = []
                for i in range(n):
                    for k in range(p):
                        for ip in range(n):
                            for j in range(m):
                                for jp in range(m):
                                    for kp in range(p):
                                        # Compute sum_c W[k*n+i,c] * U[ip*m+j,c] * V[jp*p+kp,c]
                                        val = np.sum(W_flat[k*n+i] * U_flat[ip*m+j] * V_flat[jp*p+kp])
                                        # Target
                                        if ip == i and j == 0 and jp == 0 and kp == k:
                                            target = 1
                                        elif ip == i and j == 1 and jp == 1 and kp == k:
                                            target = 1
                                        else:
                                            target = 0
                                        res.append(val - target)
                return np.array(res)
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            try:
                solution, success, info, _ = fsolve(residuals, params, full_output=True, maxfev=1000)
                if success:
                    U_sol = solution[:n*m*rank].reshape(n*m, round)
                    V_sol = solution[n*m*rank:n*m*rank+m*p*rank].reshape(m*p, rank)
                    W_sol = solution[n*m*rank+m*p*rank:].reshape(p*n, rank)
                    
                    # Verify
                    max_err = 0
                    for i in range(n):
                        for k in range(p):
                            for ip in range(n):
                                for j in range(m):
                                    for jp in range(m):
                                        for kp in range(p):
                                            val = np.sum(W_sol[k*n+i] * U_sol[ip*m+j] * V_sol[jp*p+kp])
                                            if ip == i and j == 0 and jp == 0 and kp == k:
                                                target = 1
                                            elif ip == i and j == 1 and jp == 1 and kp == k:
                                                target = 1
                                            else:
                                                target = 0
                                            max_err = max(max_err, abs(val - target))
                    
                    if max_err < 1e-8:
                        best_rank = rank
                        best_U, best_V, best_W = U_sol, V_sol, W_sol
                        break
            except Exception:
                pass
    
    # Convert to lists
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
