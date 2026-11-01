"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _find_m223_rank10(deadline):
    """Try to find a rank-10 decomposition for M_{2,2,3} using global optimization."""
    r = 10
    # U: 4x10, V: 6x10, W: 6x10
    # Total parameters: 4*10 + 6*10 + 6*10 = 160
    
    def residuals_from_params(params):
        # Unpack params
        U = params[:40].reshape(4, 10)
        V = params[40:100].reshape(6, 10)
        W = params[100:160].reshape(6, 10)
        
        res = np.zeros(12) # 2*2*3 = 12 terms
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    # U[i*2+j], V[j*3+k], W[k*2+i]
                    res[idx] = np.dot(U[i*2+j], V[j*3+k] * W[k*2+i]) - 1.0
                    idx += 1
        return res

    def objective(params):
        res = residuals_from_params(params)
        return np.sum(res**2)

    best_params = None
    best_obj = np.inf

    # Try a few random initializations
    for trial in range(10):
        if time.time() > deadline:
            break
        np.random.seed(42 + trial)
        init_params = np.random.randn(160) * 0.5
        
        # Use L-BFGS-B for speed and robustness on medium-sized problems
        result = minimize(objective, init_params, method='L-BFGS-B', 
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
            
        if best_obj < 1e-8:
            break

    if best_params is not None and best_obj < 1e-8:
        U = best_params[:40].reshape(4, 10)
        V = best_params[40:100].reshape(6, 10)
        W = best_params[100:160].reshape(6, 10)
        return (U, V, W)
        
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 100  # Leave 20s margin for safety and return
    
    # Try to find rank-10 M_{2,2,3}
    m223 = _find_m223_rank10(deadline)
    
    if m223 is not None and time.time() < deadline:
        U_m223, V_m223, W_m223 = m223
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
                        U[i * m + j][c] = float(U_m223[i * 2 + j, r])
                for j in range(2):
                    for k in range(3):
                        V[j * p + k0 + k][c] = float(V_m223[j * 3 + k, r])
                for k in range(3):
                    for i in range(2):
                        W[(k0 + k) * n + i][c] = float(W_m223[k * 2 + i, r])
        
        return {"U": U, "V": V, "W": W}
    
    # Fallback: rank-53 solution
    return _build_rank53()
# EVOLVE-BLOCK-END
