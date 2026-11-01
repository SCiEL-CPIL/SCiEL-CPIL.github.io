import time
import numpy as np
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    n, m, p = 2, 3, 4
    rank = 21
    
    # Flatten indices
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # Initialize with schoolbook (first 21 terms)
    # Schoolbook has 24 terms. We drop the last 3.
    # This is a bad initialization, but let's try random or structured.
    
    # Better initialization:
    # Use the schoolbook structure but randomize the dropped terms?
    # Or just random normal.
    
    np.random.seed(42)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    def residuals(params):
        U_flat = params[:n*m*rank].reshape(n*m, rank)
        V_flat = params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
        W_flat = params[n*m*rank + m*p*rank :].reshape(p*n, rank)
        
        res = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(m):
                        # Sum over t
                        val = 0.0
                        for t in range(rank):
                            # U[i*m+j, t] * V[l*p+k, t] * W[k*n+i, t]
                            val += U_flat[i*m+j, t] * V_flat[l*p+k, t] * W_flat[k*n+i, t]
                        target = 1.0 if j == l else 0.0
                        res.append(val - target)
        return np.array(res)
        
    # This is O(rank * n * p * m * m) per residual evaluation.
    # 21 * 2 * 4 * 3 * 3 = 1512 operations.
    # Parameters: 21 * (6 + 12 + 8) = 546.
    # This is feasible.
    
    # Use L-BFGS-B
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # To avoid getting stuck, we might try multiple restarts.
    best_res = np.inf
    best_params = x0
    
    for attempt in range(5):
        if time.time() > deadline - 10: break
        
        x0 = np.concatenate([np.random.randn(n*m, rank).flatten(), 
                             np.random.randn(m*p, rank).flatten(), 
                             np.random.randn(p*n, rank).flatten()])
        
        # Scale down to avoid overflow?
        x0 = x0 * 0.1
        
        try:
            res = minimize(residuals, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-10})
            if res.fun < best_res:
                best_res = res.fun
                best_params = res.x
        except Exception:
            pass
            
    if best_res > 1e-5:
        # Fallback to schoolbook if optimization fails
        rank = 24
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Extract best solution
    U_opt = best_params[:n*m*rank].reshape(n*m, rank)
    V_opt = best_params[n*m*rank : n*m*rank + m*p*rank].reshape(m*p, rank)
    W_opt = best_params[n*m*rank + m*p*rank :].reshape(p*n, rank)
    
    # Round to nearest half-integer
    U_rounded = np.round(U_opt * 2) / 2
    V_rounded = np.round(V_opt * 2) / 2
    W_rounded = np.round(W_opt * 2) / 2
    
    # Verify
    def check(U, V, W):
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for l in range(m):
                        val = 0.0
                        for t in range(rank):
                            val += U[i*m+j, t] * V[l*p+k, t] * W[k*n+i, t]
                        target = 1.0 if j == l else 0.0
                        if abs(val - target) > 1e-6:
                            return False
        return True
        
    if check(U_rounded, V_rounded, W_rounded):
        return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
    else:
        # Try rounding to integers
        U_int = np.round(U_opt)
        V_int = np.round(V_opt)
        W_int = np.round(W_opt)
        if check(U_int, V_int, W_int):
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
            
        # Fallback
        rank = 24
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
