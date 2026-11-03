"""Tensor decomposition for 2x3x9 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 9]

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W

def _is_decomposition_valid(U, V, W, n, m, p, tol=1e-9):
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _loss_function(params, n, m, p):
    """Compute loss for given parameter vector."""
    R = params[0]
    U = params[1:1 + R*n*m].reshape(n*m, R)
    V = params[1 + R*n*m:1 + R*n*m + R*m*p].reshape(m*p, R)
    W = params[1 + R*n*m + R*m*p:].reshape(p*n, R)
    
    total_loss = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                total_loss += (s - 1.0)**2
    return total_loss

def _optimize_decomposition(n, m, p, R, initial_params, max_iter=1000, tol=1e-12):
    """Optimize decomposition with given rank R."""
    def objective(params):
        return _loss_function(params, n, m, p)
    
    result = minimize(objective, initial_params, method='BFGS', 
                      options={'maxiter': max_iter, 'gtol': tol})
    
    R_opt = int(result.x[0])
    U = result.x[1:1 + R_opt*n*m].reshape(n*m, R_opt)
    V = result.x[1 + R_opt*n*m:1 + R_opt*n*m + R_opt*m*p].reshape(m*p, R_opt)
    W = result.x[1 + R_opt*n*m + R_opt*m*p:].reshape(p*n, R_opt)
    
    return U, V, W, result.fun

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    best_U, best_V, best_W = _build_schoolbook(n, m, p)
    best_rank = best_U.shape[1]
    
    # Try to find lower rank decompositions
    # Theoretical lower bound is max(n*m, m*p, p*n) = max(6, 27, 18) = 27
    # But actual rank could be higher. Let's try ranks from 27 up to 53.
    
    for R in range(27, min(54, best_rank)):
        if time.time() > deadline:
            break
        
        # Multiple random restarts for this rank
        found = False
        for restart in range(20):
            if time.time() > deadline:
                break
            
            # Initialize parameters
            U_init = np.random.randn(n*m, R) * 0.5
            V_init = np.random.randn(m*p, R) * 0.5
            W_init = np.random.randn(p*n, R) * 0.5
            
            # Add some structure: ensure at least one component is non-trivial
            U_init[0, 0] = 1.0
            V_init[0, 0] = 1.0
            W_init[0, 0] = 1.0
            
            params = np.concatenate([np.array([R]), U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                U_opt, V_opt, W_opt, loss = _optimize_decomposition(n, m, p, R, params, max_iter=500)
                
                # Check if we found a valid decomposition
                if _is_decomposition_valid(U_opt, V_opt, W_opt, n, m, p, tol=1e-8):
                    best_U, best_V, best_W = U_opt, V_opt, W_opt
                    best_rank = R
                    found = True
                    break
            except Exception:
                continue
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
