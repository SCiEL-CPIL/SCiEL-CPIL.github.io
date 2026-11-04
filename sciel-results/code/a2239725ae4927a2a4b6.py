import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [4, 5, 5]

def _verify(U, V, W, n, m, p, rank):
    """Verify that the decomposition is exact."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m + j] * V[j*p + k] * W[k*n + i])
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _tensor_residual(U, V, W, n, m, p, rank):
    """Compute the sum of squared residuals for the tensor identity."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    total_sq = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U[i*m + j] * V[j*p + k] * W[k*n + i])
                total_sq += (s - 1.0) ** 2
    return total_sq

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Schoolbook decomposition as fallback
    rank_school = n * m * p
    U = np.zeros((n * m, rank_school))
    V = np.zeros((m * p, rank_school))
    W = np.zeros((p * n, rank_school))
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, t] = 1
                V[j * p + k, t] = 1
                W[k * n + i, t] = 1
                t += 1
    
    best_rank = rank_school
    best_U, best_V, best_W = U, V, W
    
    # Try to find lower rank decompositions using optimization
    np.random.seed(42)
    
    for target_rank in range(best_rank - 1, max(best_rank - 50, 0), -1):
        if time.time() > deadline:
            break
        
        found = False
        for trial in range(3):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U_try = np.random.randn(n * m, target_rank) * 0.1
            V_try = np.random.randn(m * p, target_rank) * 0.1
            W_try = np.random.randn(p * n, target_rank) * 0.1
            
            def objective(params):
                U_flat = params[:n*m*target_rank].reshape(n*m, target_rank)
                V_flat = params[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
                W_flat = params[2*n*m*target_rank:].reshape(p*n, target_rank)
                return _tensor_residual(U_flat, V_flat, W_flat, n, m, p, target_rank)
            
            x0 = np.concatenate([U_try.flatten(), V_try.flatten(), W_try.flatten()])
            
            try:
                result = minimize(objective, x0, method='L-BFGS-B', 
                                  options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
                
                if result.success and result.fun < 1e-8:
                    U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
                    V_opt = result.x[n*m*target_rank:2*n*m*target_rank].reshape(m*p, target_rank)
                    W_opt = result.x[2*n*m*target_rank:].reshape(p*n, target_rank)
                    
                    if _verify(U_opt, V_opt, W_opt, n, m, p, target_rank):
                        best_rank = target_rank
                        best_U, best_V, best_W = U_opt, V_opt, W_opt
                        found = True
                        break
            except Exception:
                continue
        
        if not found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
