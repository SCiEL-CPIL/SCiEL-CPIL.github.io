import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    # Try to find a rank lower than the trivial one (120)
    # We search from 119 down to 100
    for R in range(119, 99, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Try multiple random initializations
        for attempt in range(20):
            if time.time() - start_time > time_limit:
                break
                
            # Random initialization
            U = np.random.randn(R, n, m)
            V = np.random.randn(R, m, p)
            W = np.random.randn(R, p, n)
            
            params = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
            
            def objective(params):
                U_flat = params[:R*n*m].reshape(R, n, m)
                V_flat = params[R*n*m:R*n*m + R*m*p].reshape(R, m, p)
                W_flat = params[R*n*m + R*m*p:].reshape(R, p, n)
                
                # Compute tensor T_est[i,j,k] = sum_r U[r,i,j] * V[r,j,k] * W[r,k,i]
                T_est = np.einsum('rij,rjk,rki->ijk', U_flat, V_flat, W_flat)
                residual = T_est - 1.0
                return np.sum(residual**2)
            
            # Use L-BFGS-B for optimization
            try:
                res = minimize(objective, params, method='L-BFGS-B', options={'maxiter': 1000, 'maxfun': 5000})
            except Exception:
                continue
                
            if res.fun < 1e-10:
                # Found a valid decomposition
                U_sol = res.x[:R*n*m].reshape(R, n, m).T.reshape(n*m, R)
                V_sol = res.x[R*n*m:R*n*m + R*m*p].reshape(R, m, p).T.reshape(m*p, R)
                W_sol = res.x[R*n*m + R*m*p:].reshape(R, p, n).T.reshape(p*n, R)
                
                # Convert to list
                return {"U": U_sol.tolist(), "V": V_sol.tolist(), "W": W_sol.tolist()}

    # Fallback to trivial decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
