# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _strassen_222():
    """Return Strassen decomposition for 2x2x2 multiplication."""
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    return U, V, W

def _schoolbook_221():
    """Return schoolbook decomposition for 2x2x1 multiplication."""
    U = np.eye(4)
    V = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1]], dtype=float)
    W = np.array([[1, 1, 0, 0],
                  [0, 0, 1, 1]], dtype=float)
    return U, V, W

def _residuals_m22p(params, p, rank):
    """Compute residuals for M_{2,2,p} decomposition."""
    n, m = 2, 2
    U = params[:n*m*rank].reshape(n*m, rank)
    V = params[n*m*rank:(n*m*rank + m*p*rank)].reshape(m*p, rank)
    W = params[(n*m*rank + m*p*rank):].reshape(p*n, rank)
    
    res = np.zeros(n*m*p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[idx] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - 1.0
                idx += 1
    return res

def _find_decomp_m223_rank10(deadline, max_seeds=50):
    """Search for rank-10 decomposition of M_{2,2,3}."""
    n, m, p = 2, 2, 3
    rank = 10
    total_vars = (n*m + m*p + p*n) * rank
    
    def objective(params):
        res = _residuals_m22p(params, p, rank)
        return np.sum(res**2)
    
    best_params = None
    best_obj = np.inf
    
    for seed in range(max_seeds):
        if time.time() > deadline:
            break
        np.random.seed(seed)
        x0 = np.random.randn(total_vars) * 0.5
        try:
            # First pass: L-BFGS-B
            res1 = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            # Second pass: Nelder-Mead polish
            res2 = minimize(res1.x, objective, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
            
            if res2.fun < best_obj:
                best_obj = res2.fun
                best_params = res2.x
            if best_obj < 1e-10:
                break
        except Exception:
            continue
    
    if best_params is None or best_obj > 1e-8:
        return None
    
    U = best_params[:n*m*rank].reshape(n*m, rank)
    V = best_params[n*m*rank:(n*m*rank + m*p*rank)].reshape(m*p, rank)
    W = best_params[(n*m*rank + m*p*rank):].reshape(p*n, rank)
    
    # Snap to half-integers or integers
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Verify
    res = _residuals_m22p(np.concatenate([U.flatten(), V.flatten(), W.flatten()]), p, rank)
    if np.max(np.abs(res)) < 1e-6:
        return U, V, W
    return None

def _build_rank53():
    """Build the guaranteed rank-53 decomposition (7 Strassen blocks + 1 schoolbook)."""
    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen, V_strassen, W_strassen = _strassen_222()
    U_school, V_school, W_school = _schoolbook_221()
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = int(U_strassen[i * 2 + j, r])
            for j in range(2):
                V[j * p + k0][c] = int(V_strassen[j * 2, r])
                V[j * p + k1][c] = int(V_strassen[j * 2 + 1, r])
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = int(W_strassen[k * 2 + i, r])
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = int(U_school[i * 2 + j, r])
        for j in range(2):
            V[j * p + k_last][c] = int(V_school[j, r])
        for i in range(2):
            W[k_last * n + i][c] = int(W_school[i, r])
    
    return {"U": U, "V": V, "W": W}

def _build_rank50(U3, V3, W3):
    """Build rank-50 decomposition using 5 blocks of M_{2,2,3}."""
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block in range(5):
        k_start = 3 * block
        col_offset = 10 * block
        for r in range(10):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = float(U3[i * 2 + j, r])
            for j in range(2):
                for k in range(3):
                    V[j * p + k_start + k][c] = float(V3[j * 3 + k, r])
            for k in range(3):
                for i in range(2):
                    W[(k_start + k) * n + i][c] = float(W3[k * 2 + i, r])
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Fallback: rank-53 solution
    fallback = _build_rank53()
    
    # Try to find rank-10 decomposition for M_{2,2,3}
    m223 = _find_decomp_m223_rank10(deadline, max_seeds=50)
    
    if m223 is not None:
        U3, V3, W3 = m223
        return _build_rank50(U3, V3, W3)
    
    # Fallback to rank-53
    return fallback
# EVOLVE-BLOCK-END
