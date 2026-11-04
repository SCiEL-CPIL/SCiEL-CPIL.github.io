import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer
    
    n, m, p = DIMENSIONS
    # Target tensor is 1 everywhere
    # Shapes:
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    best_rank = n * m * p
    best_U = [[0] * best_rank for _ in range(n * m)]
    best_V = [[0] * best_rank for _ in range(m * p)]
    best_W = [[0] * best_rank for _ in range(p * n)]
    # Initialize with schoolbook
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j][col] = 1
                best_V[j*p+k][col] = 1
                best_W[k*n+i][col] = 1
                col += 1

    # Try to find lower rank
    for R in range(15, 25): # Try ranks 15 to 24
        if time.time() > deadline - 20: break # Leave 20s for finishing
        
        for restart in range(10): # 10 restarts per rank
            if time.time() > deadline - 5: break
            
            # Random initialization
            np.random.seed(restart + R)
            U = np.random.randn(n*m, R) * 0.1
            V = np.random.randn(m*p, R) * 0.1
            W = np.random.randn(p*n, R) * 0.1
            
            # Flatten for optimization
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def objective(x):
                U_flat = x[:n*m*R].reshape(n*m, R)
                V_flat = x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W_flat = x[n*m*R + m*p*R:].reshape(p*n, R)
                
                error = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_flat[i*m+j], V_flat[j*p+k] * W_flat[k*n+i])
                            error += (val - 1.0)**2
                return error
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
                
                if res.fun < 1e-6:
                    # Potential solution
                    U_sol = res.x[:n*m*R].reshape(n*m, R)
                    V_sol = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                    W_sol = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                    
                    # Round to nearest 0.5
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    # Verify exactness
                    is_exact = True
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.dot(U_rounded[i*m+j], V_rounded[j*p+k] * W_rounded[k*n+i])
                                if not np.isclose(val, 1.0, atol=1e-9):
                                    is_exact = False
                                    break
                        if not is_exact: break
                    if not is_exact: break
                    
                    if is_exact:
                        # Found a valid decomposition of rank R
                        return {
                            "U": U_rounded.tolist(),
                            "V": V_rounded.tolist(),
                            "W": W_rounded.tolist()
                        }
            except Exception:
                pass

    # Return schoolbook if no better found
    return {"U": best_U, "V": best_V, "W": best_W}
