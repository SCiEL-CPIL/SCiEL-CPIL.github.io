import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 8]

def _verify(U, V, W, n, m, p, rank):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    U_f = np.array(U, dtype=float)
    V_f = np.array(V, dtype=float)
    W_f = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_f[i*m+j, r] * V_f[j*p+k, r] * W_f[k*n+i, r]
                if abs(val - 1.0) > 1e-9:
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

def _find_decomposition(n, m, p, target_rank, max_time=10):
    n_m = n * m
    m_p = m * p
    p_n = p * n
    
    def error(params):
        U = params[:n_m*target_rank].reshape(n_m, target_rank)
        V = params[n_m*target_rank:n_m*target_rank + m_p*target_rank].reshape(m_p, target_rank)
        W = params[n_m*target_rank + m_p*target_rank:].reshape(p_n, target_rank)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        return err

    # Initialize with small random values
    np.random.seed(42)
    x0 = np.random.randn(n_m*target_rank + m_p*target_rank + p_n*target_rank) * 0.1
    
    start_time = time.time()
    best_result = None
    best_err = float('inf')
    
    # Try a few random starts
    for trial in range(5):
        if time.time() - start_time > max_time:
            break
        x0 = np.random.randn(n_m*target_rank + m_p*target_rank + p_n*target_rank) * 0.1
        res = minimize(error, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_err:
            best_err = res.fun
            best_result = res.x
            
    if best_err < 1e-5 and best_result is not None:
        U = best_result[:n_m*target_rank].reshape(n_m, target_rank)
        V = best_result[n_m*target_rank:n_m*target_rank + m_p*target_rank].reshape(m_p, target_rank)
        W = best_result[n_m*target_rank + m_p*target_rank:].reshape(p_n, target_rank)
        
        # Snap to half-integers
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        if _verify(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist(), n, m, p, target_rank):
            return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
            
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Try to find a better decomposition
    for target_rank in [50, 60, 65]:
        if time.time() > deadline:
            break
        result = _find_decomposition(n, m, p, target_rank, max_time=30)
        if result:
            return result
            
    return _schoolbook(n, m, p)
