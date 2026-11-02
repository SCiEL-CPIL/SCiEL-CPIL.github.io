import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline with margin
    
    # Build the tensor equations
    # For each (i,j,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # i in [0,n), j in [0,m), k in [0,p)
    
    # Try ranks from low to high
    # Lower bound: max(n*m, m*p, n*p) = max(9,12,12) = 12
    # Try starting from 21 (known for 3x3x3) and go up
    
    best_result = None
    
    for rank in range(21, 37):
        if time.time() > deadline - 30:
            break
        
        # Initialize
        np.random.seed(rank)
        U = np.random.randn(n*m, rank) * 0.1
        V = np.random.randn(m*p, rank) * 0.1
        W = np.random.randn(p*n, rank) * 0.1
        
        def objective(x):
            U_flat = x[:n*m*rank].reshape(n*m, rank)
            V_flat = x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_flat = x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            errors = []
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U_flat[i*m+j, :] * V_flat[j*p+k, :] * W_flat[k*n+i, :])
                        errors.append(val - 1.0)
            errors = np.array(errors)
            return np.sum(errors**2)
        
        x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', 
                             options={'maxiter': 5000, 'ftol': 1e-20, 'gtol': 1e-20})
            
            if result.fun < 1e-10:
                U_sol = result.x[:n*m*rank].reshape(n*m, rank)
                V_sol = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                W_sol = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                # Try to round to integers or half-integers
                # First try integers
                U_int = np.round(U_sol)
                V_int = np.round(V_sol)
                W_int = np.round(W_sol)
                
                # Check if integer solution is exact
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_int[i*m+j, :] * V_int[j*p+k, :] * W_int[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    best_result = {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                    break
                
                # Try half-integers
                U_half = np.round(U_sol * 2) / 2
                V_half = np.round(V_sol * 2) / 2
                W_half = np.round(W_sol * 2) / 2
                
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_half[i*m+j, :] * V_half[j*p+k, :] * W_half[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-6:
                    best_result = {
                        "U": [[float(v) for v in row] for row in U_half],
                        "V": [[float(v) for v in row] for row in V_half],
                        "W": [[float(v) for v in row] for row in W_half]
                    }
                    break
                    
        except Exception:
            continue
    
    if best_result is None:
        # Fallback: schoolbook decomposition
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
        best_result = {"U": U, "V": V, "W": W}
    
    return best_result
