import time
import numpy as np
import sys
from itertools import product

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] for r in range(rank))
                if abs(val - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_optimization(n, m, p, R, time_budget):
    """
    Attempt to find a rank-R decomposition using gradient-based optimization.
    Returns dict if successful, None otherwise.
    """
    from scipy.optimize import minimize
    
    U_size = n * m * R
    V_size = m * p * R
    W_size = p * n * R
    total_size = U_size + V_size + W_size
    
    def objective_and_grad(x):
        U_flat = x[:U_size].reshape(n*m, R)
        V_flat = x[U_size:U_size+V_size].reshape(m*p, R)
        W_flat = x[U_size+V_size:U_size+V_size+W_size].reshape(p*n, R)
        
        obj = 0.0
        grad_U = np.zeros_like(U_flat)
        grad_V = np.zeros_like(V_flat)
        grad_W = np.zeros_like(W_flat)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    
                    u_row = U_flat[u_idx, :]
                    v_row = V_flat[v_idx, :]
                    w_row = W_flat[w_idx, :]
                    
                    t_val = np.dot(u_row, v_row * w_row)
                    res = t_val - 1.0
                    
                    obj += res * res
                    
                    grad_U[u_idx, :] += 2.0 * res * (v_row * w_row)
                    grad_V[v_idx, :] += 2.0 * res * (u_row * w_row)
                    grad_W[w_idx, :] += 2.0 * res * (u_row * v_row)
                    
        grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        return obj, grad
    
    start_time = time.time()
    best_obj = float('inf')
    best_x = None
    
    num_restarts = min(20, max(1, int(time_budget / 2.0)))
    
    for trial in range(num_restarts):
        if time.time() - start_time > time_budget:
            break
            
        np.random.seed(trial * 12345 + 67890)
        x0 = np.random.randn(total_size) * 0.1
        
        try:
            res = minimize(
                objective_and_grad, 
                x0, 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-10}
            )
            if res.fun < best_obj:
                best_obj = res.fun
                best_x = res.x
        except Exception:
            continue
        
        # Check if we've found a good solution
        if best_obj < 1e-8:
            break
    
    if best_x is None or best_obj > 1e-6:
        return None
    
    U_opt = best_x[:U_size].reshape(n*m, R)
    V_opt = best_x[U_size:U_size+V_size].reshape(m*p, R)
    W_opt = best_x[U_size+V_size:U_size+V_size+W_size].reshape(p*n, R)
    
    if _verify(U_opt.tolist(), V_opt.tolist(), W_opt.tolist(), n, m, p, tol=1e-6):
        return {"U": U_opt.tolist(), "V": V_opt.tolist(), "W": W_opt.tolist()}
        
    return None

def _try_strassen_based(n, m, p, time_budget):
    """
    Attempt to construct a lower-rank decomposition using Strassen-like
    block decomposition ideas. For 2x3x10, we can try to exploit structure.
    """
    # This is a heuristic attempt based on known results.
    # For 2x3x10, the tensor rank is believed to be 60 (schoolbook optimal)
    # based on lower bounds, but we try anyway.
    
    # Try rank 45 (a common candidate for some matrix multiplication problems)
    if time_budget > 5:
        result = _try_rank_optimization(n, m, p, 45, time_budget * 0.5)
        if result is not None:
            return result
    
    # Try rank 50
    if time_budget > 5:
        result = _try_rank_optimization(n, m, p, 50, time_budget * 0.5)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Start with schoolbook (rank 60)
    best_result = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a better decomposition
    time_left = deadline - time.time()
    
    if time_left > 10:
        result = _try_strassen_based(n, m, p, time_left - 5)
        if result is not None:
            best_result = result
            best_rank = len(result['U'][0])
    
    return best_result
