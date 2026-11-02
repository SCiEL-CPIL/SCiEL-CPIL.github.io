"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _try_m223_rank10(timeout=80):
    """
    Attempt to find a rank-10 decomposition for M_{2,2,3}.
    Returns (U, V, W) as lists or None.
    """
    start = time.time()
    deadline = start + timeout
    
    # Tensor entries: for each (i,j,k), the bilinear form must equal 1
    # U[i*2+j, r] * V[j*3+k, r] * W[k*2+i, r] summed over r = 1
    # Total constraints: 2*2*3 = 12
    
    def residuals(params):
        U = params[:40].reshape(4, 10)
        V = params[40:100].reshape(6, 10)
        W = params[100:160].reshape(6, 10)
        res = np.zeros(12)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    res[idx] = np.dot(U[i*2+j], V[j*3+k] * W[k*2+i]) - 1.0
                    idx += 1
        return res
    
    def objective(params):
        res = residuals(params)
        return np.sum(res**2)
    
    best_params = None
    best_obj = np.inf
    
    # Try multiple seeds with different initializations
    for seed in range(20):
        if time.time() > deadline:
            break
        np.random.seed(seed)
        
        # Different initialization strategies
        if seed < 5:
            x0 = np.random.randn(160) * 0.5
        elif seed < 10:
            # Initialize with small structured values
            x0 = np.zeros(160)
            x0[:40] = np.random.randn(40) * 0.3
            x0[40:100] = np.random.randn(60) * 0.3
            x0[100:160] = np.random.randn(60) * 0.3
        else:
            # Sparse-like initialization
            x0 = np.random.randn(160)
            mask = np.random.rand(160) < 0.3
            x0[~mask] = 0
            x0[mask] *= 2
        
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x.copy()
            
        if best_obj < 1e-14:
            break
    
    if best_params is None or best_obj > 1e-10:
        return None
    
    U = best_params[:40].reshape(4, 10)
    V = best_params[40:100].reshape(6, 10)
    W = best_params[100:160].reshape(6, 10)
    
    # Snap to half-integers
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Verify after snapping
    res = residuals(np.concatenate([U.flatten(), V.flatten(), W.flatten()]))
    if np.max(np.abs(res)) < 1e-6:
        return U.tolist(), V.tolist(), W.tolist()
    
    # Try snapping to integers only
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    res_int = residuals(np.concatenate([U_int.flatten(), V_int.flatten(), W_int.flatten()]))
    if np.max(np.abs(res_int)) < 1e-6:
        return U_int.tolist(), V_int.tolist(), W_int.tolist()
    
    return None

def _build_rank53():
    """Build the guaranteed rank-53 decomposition."""
    n, m, p = DIMENSIONS
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    U_school = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    V_school = [[1,0,1,0],[0,1,0,1]]
    W_school = [[1,1,0,0],[0,0,1,1]]
    
    col_offset = 0
    for block in range(7):
        k0 = 2 * block
        k1 = 2 * block + 1
        for r in range(7):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_strassen[i * 2 + j][r]
            for j in range(2):
                V[j * p + k0][c] = V_strassen[j * 2][r]
                V[j * p + k1][c] = V_strassen[j * 2 + 1][r]
            for k in range(2):
                kk = k0 + k
                for i in range(2):
                    W[kk * n + i][c] = W_strassen[k * 2 + i][r]
        col_offset += 7
    
    k_last = 14
    for r in range(4):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U_school[i * 2 + j][r]
        for j in range(2):
            V[j * p + k_last][c] = V_school[j][r]
        for i in range(2):
            W[k_last * n + i][c] = W_school[i][r]
    
    return {"U": U, "V": V, "W": W}

def _build_rank50(U_m223, V_m223, W_m223):
    """Build rank-50 decomposition using 5 blocks of M_{2,2,3}."""
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for block in range(5):
        k0 = 3 * block
        col_offset = 10 * block
        for r in range(10):
            c = col_offset + r
            for i in range(2):
                for j in range(2):
                    U[i * m + j][c] = U_m223[i * 2 + j][r]
            for j in range(2):
                for k in range(3):
                    V[j * p + k0 + k][c] = V_m223[j * 3 + k][r]
            for k in range(3):
                for i in range(2):
                    W[(k0 + k) * n + i][c] = W_m223[k * 2 + i][r]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Try to find rank-10 M_{2,2,3} decomposition
    m223_result = _try_m223_rank10(timeout=100)
    
    if m223_result is not None and time.time() < deadline:
        U_m223, V_m223, W_m223 = m223_result
        return _build_rank50(U_m223, V_m223, W_m223)
    
    # Fallback to guaranteed rank-53
    return _build_rank53()
# EVOLVE-BLOCK-END
