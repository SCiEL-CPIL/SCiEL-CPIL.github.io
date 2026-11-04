import time
import sys
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify that U, V, W form a valid exact decomposition with integer/half-integer entries."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check entries are integer or half-integer
    for r in range(rank):
        for i in range(n * m):
            val = U_arr[i, r]
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
        for j in range(m * p):
            val = V_arr[j, r]
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False
        for k in range(p * n):
            val = W_arr[k, r]
            if abs(val * 2 - round(val * 2)) > 1e-9:
                return False

    # Check tensor equation
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _objective(params, n, m, p, rank):
    U = params[:n * m * rank].reshape(n * m, rank)
    V = params[n * m * rank:n * m * rank + m * p * rank].reshape(m * p, rank)
    W = params[n * m * rank + m * p * rank:].reshape(p * n, rank)
    
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                error += (s - 1.0) ** 2
    return error

def _find_decomposition(n, m, p, target_rank, max_iter=1000, timeout=10):
    start_time = time.time()
    U_size = n * m * target_rank
    V_size = m * p * target_rank
    W_size = p * n * target_rank
    total_size = U_size + V_size + W_size
    
    # Initialize with random small values
    random.seed(42)
    params = np.array([random.uniform(-1, 1) for _ in range(total_size)])
    
    # Use Nelder-Mead for optimization
    # Note: This is a non-convex problem, so we do multiple restarts
    best_params = None
    best_err = float('inf')
    
    for restart in range(5):
        if time.time() - start_time > timeout:
            break
        if restart > 0:
            params = np.array([random.uniform(-1, 1) for _ in range(total_size)])
        
        result = minimize(_objective, params, args=(n, m, p, target_rank), method='Nelder-Mead', options={'maxiter': max_iter, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if result.fun < best_err:
            best_err = result.fun
            best_params = result.x
            
        if best_err < 1e-6:
            break
            
    if best_err < 1e-6:
        U = best_params[:U_size].reshape(n * m, target_rank)
        V = best_params[U_size:U_size + V_size].reshape(m * p, target_rank)
        W = best_params[U_size + V_size:].reshape(p * n, target_rank)
        
        # Round to nearest half-integer
        U = np.round(U * 2) / 2
        V = np.round(V * 2) / 2
        W = np.round(W * 2) / 2
        
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    # We try ranks from 41 down to 30
    # This is a heuristic search; it may not find a solution, but it's worth trying
    for target_rank in range(best_rank - 1, 29, -1):
        if time.time() > deadline - 10:
            break
            
        # We limit the search time per rank to avoid exceeding the deadline
        result = _find_decomposition(n, m, p, target_rank, max_iter=200, timeout=15)
        
        if result is not None:
            best = result
            best_rank = target_rank
            # If we find a low rank, we can stop or continue to find even lower
            # For now, we continue to see if we can go lower, but we break if time is tight
            if best_rank < 35:
                break
                
    return best
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    res = construct()
