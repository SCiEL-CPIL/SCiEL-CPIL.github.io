import numpy as np
from scipy.optimize import least_squares
import time

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Check shapes
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # Check exact decomposition
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if not np.isclose(val, 1.0, atol=1e-9):
                    return False
                    
    # Check half-integer property
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if not np.isclose(val * 2, np.round(val * 2), atol=1e-9):
                    return False
    return True

def _schoolbook():
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

def _try_find_rank(r, time_limit=10):
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    total_vars = (size_u + size_v + size_w) * r
    
    def residuals(x):
        U = x[:size_u * r].reshape(size_u, r)
        V = x[size_u * r : size_u * r + size_v * r].reshape(size_v, r)
        W = x[size_u * r + size_v * r :].reshape(size_w, r)
        
        res = np.empty(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[idx] = np.sum(U[i*m + j] * V[j*p + k] * W[k*n + i]) - 1.0
                    idx += 1
        return res
    
    # Try random starts
    for attempt in range(100):
        if time.time() - start_time > time_limit:
            break
        
        x0 = np.random.normal(0, 1, total_vars)
        
        try:
            sol = least_squares(residuals, x0, method='lm', max_nfev=2000)
            if sol.cost < 1e-10:
                U = sol.x[:size_u * r].reshape(size_u, r)
                V = sol.x[size_u * r : size_u * r + size_v * r].reshape(size_v, r)
                W = sol.x[size_u * r + size_v * r :].reshape(size_w, r)
                
                # Snap to half-integers
                U_snapped = np.round(U * 2) / 2
                V_snapped = np.round(V * 2) / 2
                W_snapped = np.round(W * 2) / 2
                
                # Convert to lists for verification and return
                U_list = U_snapped.tolist()
                V_list = V_snapped.tolist()
                W_list = W_snapped.tolist()
                
                if _verify(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
        except Exception:
            pass
            
    return None

def construct():
    # Deadline guard
    start = time.time()
    limit = 110.0 # Leave 10s margin
    
    # Try rank 17
    if time.time() - start < limit:
        result = _try_find_rank(17, time_limit=min(30, limit - (time.time() - start)))
        if result:
            return result
            
    # Try rank 16
    if time.time() - start < limit:
        result = _try_find_rank(16, time_limit=min(30, limit - (time.time() - start)))
        if result:
            return result
            
    # Fall back to schoolbook
    return _schoolbook()
