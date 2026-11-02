"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _try_m225_rank15(timeout=80):
    """
    Attempt to find a rank-15 decomposition for M_{2,2,5}.
    Returns (U, V, W) as numpy arrays or None.
    """
    start = time.time()
    deadline = start + timeout
    
    # Tensor: U[i*2+j, r] * V[j*5+k, r] * W[k*2+i, r] summed over r = delta
    # Constraints: 2*2*5 = 20
    # Variables: 4*15 + 10*15 + 10*15 = 60+150+150 = 360
    
    def residuals(params):
        U = params[:60].reshape(4, 15)
        V = params[60:210].reshape(10, 15)
        W = params[210:360].reshape(10, 15)
        res = np.zeros(20)
        idx = 0
        for i in range(2):
            for j in range(2):
                for k in range(5):
                    res[idx] = np.dot(U[i*2+j], V[j*5+k] * W[k*2+i]) - 1.0
                    idx += 1
        return res
    
    def objective(params):
        res = residuals(params)
        return np.sum(res**2)
    
    best_params = None
    best_obj = np.inf
    
    # Try multiple seeds with different initializations
    for seed in range(30):
        if time.time() > deadline:
            break
        np.random.seed(seed)
        
        # Different initialization strategies
        if seed < 10:
            x0 = np.random.randn(360) * 0.5
        elif seed < 20:
            # Sparse-like initialization
            x0 = np.random.randn(360)
            mask = np.random.rand(360) < 0.2
            x0[~mask] = 0
            x0[mask] *= 2
        else:
            # Small structured values
            x0 = np.random.randn(360) * 0.3
        
        res = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x.copy()
            
        if best_obj < 1e-14:
            break
    
    if best_params is None or best_obj > 1e-10:
        return None
    
    U = best_params[:60].reshape(4, 15)
    V = best_params[60:210].reshape(10, 15)
    W = best_params[210:360].reshape(10, 15)
    
    # Snap to half-integers
    U_snap = np.round(U * 2) / 2
    V_snap = np.round(V * 2) / 2
    W_snap = np.round(W * 2) / 2
    
    # Verify after snapping
    res = residuals(np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()]))
    if np.max(np.abs(res)) < 1e-6:
        return U_snap, V_snap, W_snap
    
    # Try snapping to integers only
    U_int = np.round(U)
    V_int = np.round(V)
    W_int = np.round(W)
    res_int = residuals(np.concatenate([U_int.flatten(), V_int.flatten(), W_int.flatten()]))
    if np.max(np.abs(res_int)) < 1e-6:
        return U_int, V_int, W_int
    
    # Try snapping to quarter-integers
    U_quarter = np.round(U * 4) / 4
    V_quarter = np.round(V * 4) / 4
    W_quarter = np.round(W * 4) / 4
    res_quarter = residuals(np.concatenate([U_quarter.flatten(), V_quarter.flatten(), W_quarter.flatten()]))
    if np.max(np.abs(res_quarter)) < 1e-6:
        return U_quarter, V_quarter, W_quarter
    
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

def _build_rank50_m225(U_m225, V_m225, W_m225):
    """
    Build rank-50 decomposition:
    - Columns 0-9: five 2x2x2 Strassen blocks (rank 35)
    - Columns 10-14: one 2x2x5 block (rank 15)
    """
    n, m, p = DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 blocks
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V_strassen = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Place 5 Strassen blocks for columns 0-9
    col_offset = 0
    for block in range(5):
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
    
    # Place 2x2x5 block for columns 10-14
    k0 = 10
    for r in range(15):
        c = col_offset + r
        for i in range(2):
            for j in range(2):
                U[i * m + j][c] = U_m225[i * 2 + j][r]
        for j in range(2):
            for k in range(5):
                V[j * p + k0 + k][c] = V_m225[j * 5 + k][r]
        for k in range(5):
            for i in range(2):
                W[(k0 + k) * n + i][c] = W_m225[k * 2 + i][r]
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Try to find rank-15 M_{2,2,5} decomposition
    m225_result = _try_m225_rank15(timeout=100)
    
    if m225_result is not None and time.time() < deadline:
        U_m225, V_m225, W_m225 = m225_result
        return _build_rank50_m225(U_m225, V_m225, W_m225)
    
    # Fallback to guaranteed rank-53
    return _build_rank53()
# EVOLVE-BLOCK-END
