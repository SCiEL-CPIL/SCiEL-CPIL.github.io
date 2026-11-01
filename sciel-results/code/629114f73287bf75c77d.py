import numpy as np
import time
import sys
from scipy.optimize import least_squares

DIMENSIONS = [3, 3, 4]

def _schoolbook(n, m, p):
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

def _check_and_round(U, V, W, tol=1e-5):
    """Check if U, V, W are close to half-integers and return rounded versions if so."""
    Uf = U.flatten()
    Vf = V.flatten()
    Wf = W.flatten()
    
    dist_U = np.abs(Uf - np.round(2 * Uf) / 2.0)
    dist_V = np.abs(Vf - np.round(2 * Vf) / 2.0)
    dist_W = np.abs(Wf - np.round(2 * Wf) / 2.0)
    
    if np.max(dist_U) < tol and np.max(dist_V) < tol and np.max(dist_W) < tol:
        U_r = np.round(2 * Uf) / 2.0
        V_r = np.round(2 * Vf) / 2.0
        W_r = np.round(2 * Wf) / 2.0
        
        n, m, p = DIMENSIONS
        U_3d = U_r.reshape(n*m, -1)
        V_3d = V_r.reshape(m*p, -1)
        W_3d = W_r.reshape(p*n, -1)
        
        rank = U_3d.shape[1]
        
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U_3d[i*m+j, r] * V_3d[j*p+k, r] * W_3d[k*n+i, r]
                    if abs(val - 1.0) > tol:
                        error = 1.0
                        break
                if error: break
            if error: break
        
        if error == 0.0:
            return True, U_r.reshape(U.shape), V_r.reshape(V.shape), W_r.reshape(W.shape)
        
    return False, None, None, None

def _try_find_decomposition(n, m, p, rank, max_nfev=500):
    size_U = n * m * rank
    size_V = m * p * rank
    size_W = p * n * rank
    total_size = size_U + size_V + size_W
    
    def residual(x):
        Uf = x[:size_U].reshape(n*m, rank)
        Vf = x[size_U:size_U+size_V].reshape(m*p, rank)
        Wf = x[size_U+size_V:].reshape(p*n, rank)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = Uf[i*m+j, :]
                    v_row = Vf[j*p+k, :]
                    w_row = Wf[k*n+i, :]
                    val = np.dot(u_row, v_row * w_row)
                    res[idx] = val - 1.0
                    idx += 1
        return res
    
    # Use a deterministic but varied initialization strategy
    # Start with small random noise
    for seed in [42, 123, 999]:
        np.random.seed(seed)
        x0 = np.random.normal(0, 0.1, total_size)
        
        try:
            res = least_squares(residual, x0, method='lm', xtol=1e-10, ftol=1e-10, max_nfev=max_nfev)
            
            if res.success or res.cost < 1e-8:
                x_sol = res.x
                U_sol = x_sol[:size_U].reshape(n*m, rank)
                V_sol = x_sol[size_U:size_U+size_V].reshape(m*p, rank)
                W_sol = x_sol[size_U+size_V:].reshape(p*n, rank)
                
                valid, U_r, V_r, W_r = _check_and_round(U_sol, V_sol, W_sol)
                if valid:
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
                    
        except Exception as e:
            pass
        
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find a better decomposition
    # Start with higher ranks that are more likely to have solutions, then go lower
    # The schoolbook rank is 36. We try to find lower ranks.
    # Known bounds suggest R(3,3,4) is likely around 30-33.
    
    # We will iterate through ranks.
    # To be safe and fast, we limit the number of attempts per rank.
    
    # Ranks to try: 35, 34, 33, 32, 31, 30
    # If we find a solution, we return it.
    
    ranks_to_try = [35, 34, 33, 32, 31, 30]
    
    for rank in ranks_to_try:
        if time.time() > deadline - 5: # Leave 5s for safety
            break
            
        result = _try_find_decomposition(n, m, p, rank)
        if result:
            return result
            
    # Fallback
    return _schoolbook(n, m, p)
