import time
import random
import sys
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 3]

def _verify(U, V, W):
    """Verify that U,V,W with given indices correctly decompose matrix multiplication.
    Returns True if decomposition is exact."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _construct_schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def _objective(params, rank):
    n, m, p = 3, 3, 3
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    
    U_flat = params[:size_u]
    V_flat = params[size_u:size_u + size_v]
    W_flat = params[size_u + size_v:]
    
    U = U_flat.reshape(n * m, rank)
    V = V_flat.reshape(m * p, rank)
    W = W_flat.reshape(p * n, rank)
    
    # Compute error
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                error += (s - 1.0) ** 2
    return error

def _search_rank_opt(target_rank, deadline, seed=42):
    n, m, p = 3, 3, 3
    rank = target_rank
    random.seed(seed)
    np.random.seed(seed)
    
    size_u = n * m * rank
    size_v = m * p * rank
    size_w = p * n * rank
    total_size = size_u + size_v + size_w
    
    # Initialize with small random values
    x0 = np.random.uniform(-1, 1, total_size)
    
    start_time = time.time()
    
    best_result = None
    best_error = float('inf')
    
    # Try multiple methods and seeds
    methods = ['Powell', 'Nelder-Mead']
    seeds = [1, 2, 3, 4, 5]
    
    for method in methods:
        for s in seeds:
            if time.time() > deadline - 10:
                break
            np.random.seed(s + seed)
            x0 = np.random.uniform(-1, 1, total_size)
            try:
                if method == 'Powell':
                    res = minimize(_objective, x0, args=(rank,), method='Powell', 
                                   options={'maxiter': 50, 'ftol': 1e-4, 'xtol': 1e-4})
                else:
                    res = minimize(_objective, x0, args=(rank,), method='Nelder-Mead', 
                                   options={'maxiter': 500, 'fatol': 1e-4, 'xatol': 1e-4})
                
                if res.fun < best_error:
                    best_error = res.fun
                    best_result = res.x
                    
                    if best_error < 1e-3:
                        U_flat = best_result[:size_u]
                        V_flat = best_result[size_u:size_u + size_v]
                        W_flat = best_result[size_u + size_v:]
                        
                        U = (U_flat.reshape(n * m, rank) * 2).round() / 2
                        V = (V_flat.reshape(m * p, rank) * 2).round() / 2
                        W = (W_flat.reshape(p * n, rank) * 2).round() / 2
                        
                        U_list = U.tolist()
                        V_list = V.tolist()
                        W_list = W.tolist()
                        
                        if _verify(U_list, V_list, W_list):
                            return {"U": U_list, "V": V_list, "W": W_list}
                            
            except Exception:
                pass
                
    return None

def construct():
    overall_deadline = time.time() + 100
    
    # Try to find a rank-21 decomposition
    start = time.time()
    r21 = _search_rank_opt(21, overall_deadline - 10, seed=1)
    if r21 is not None:
        return r21
        
    # Fall back to schoolbook rank 27
    return _construct_schoolbook()
