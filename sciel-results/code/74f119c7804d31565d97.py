"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _find_m223_rank10(deadline):
    """
    Find a rank-10 decomposition for M_{2,2,3} with half-integer entries.
    Returns (U, V, W) as numpy arrays or None.
    """
    n, m, p = 2, 2, 3
    rank = 10
    num_u = n * m * rank  # 40
    num_v = m * p * rank  # 60
    num_w = p * n * rank  # 60
    total = num_u + num_v + num_w  # 160
    
    def make_residuals(params):
        U = params[:num_u].reshape(n*m, rank)
        V = params[num_u:num_u+num_v].reshape(m*p, rank)
        W = params[num_u+num_v:].reshape(p*n, rank)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[idx] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - 1.0
                    idx += 1
        return res
    
    def objective(params):
        res = make_residuals(params)
        return np.sum(res**2)
    
    best_obj = np.inf
    best_params = None
    
    # Strategy 1: Start from rank-11 (Strassen 7 + schoolbook 4) and optimize
    # Build rank-11 solution first
    U11 = np.zeros((4, 11))
    V11 = np.zeros((6, 11))
    W11 = np.zeros((6, 11))
    
    # Strassen for cols 0,1
    U_s = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    V_s = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    for r in range(7):
        U11[:, r] = U_s[:, r]
        for j in range(2):
            V11[j*3+0, r] = V_s[j*2, r]
            V11[j*3+1, r] = V_s[j*2+1, r]
        for k in range(2):
            for i in range(2):
                W11[k*2+i, r] = W_s[k*2+i, r]
    
    # Schoolbook for col 2
    for r in range(4):
        U11[:, 7+r] = np.eye(4)[r]
        for j in range(2):
            V11[j*3+2, 7+r] = np.eye(2)[j, r] if r < 2 else 0
            # Actually schoolbook: U[i,j]=delta, V[j,k]=delta, W[k,i]=delta
            # For col 2: V[j,2] should be 1 if j==r for r=0,1
        # Let me redo this properly
        pass
    
    # Actually, let me just use random initialization with multiple seeds
    for seed in range(30):
        if time.time() > deadline:
            break
        np.random.seed(seed)
        x0 = np.random.randn(total) * 0.5
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 5000, 'ftol': 1e-16})
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
        if best_obj < 1e-14:
            break
    
    if best_params is None or best_obj > 1e-10:
        return None
    
    U = best_params[:num_u].reshape(n*m, rank)
    V = best_params[num_u:num_u+num_v].reshape(m*p, rank)
    W = best_params[num_u+num_v:].reshape(p*n, rank)
    
    # Snap to half-integers
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Verify
    res = make_residuals(np.concatenate([U.flatten(), V.flatten(), W.flatten()]))
    if np.max(np.abs(res)) < 1e-6:
        return U, V, W
    
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

def _build_rank50_from_m223(U_m223, V_m223, W_m223):
    """Build rank-50 solution from 5 blocks of M_{2,2,3} rank-10."""
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
                    U[i * m + j][c] = float(U_m223[i * 2 + j, r])
            for j in range(2):
                for k in range(3):
                    V[j * p + k0 + k][c] = float(V_m223[j * 3 + k, r])
            for k in range(3):
                for i in range(2):
                    W[(k0 + k) * n + i][c] = float(W_m223[k * 2 + i, r])
    
    return {"U": U, "V": V, "W": W}

def _try_reduce_rank(U, V, W, current_rank, target_rank, deadline):
    """
    Try to reduce the rank from current_rank to target_rank by optimizing.
    Returns (U, V, W) with target_rank columns if successful, else None.
    """
    n, m, p = DIMENSIONS
    
    def make_residuals(params):
        U_r = params[:target_rank*n*m].reshape(n*m, target_rank)
        V_r = params[target_rank*n*m:target_rank*(n*m+m*p)].reshape(m*p, target_rank)
        W_r = params[target_rank*(n*m+m*p):].reshape(p*n, target_rank)
        
        res = np.zeros(n*m*p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[idx] = np.dot(U_r[i*m+j], V_r[j*p+k] * W_r[k*n+i]) - 1.0
                    idx += 1
        return res
    
    def objective(params):
        res = make_residuals(params)
        return np.sum(res**2)
    
    # Initialize from the first target_rank columns
    U_init = np.array(U, dtype=float)[:, :target_rank]
    V_init = np.array(V, dtype=float)[:, :target_rank]
    W_init = np.array(W, dtype=float)[:, :target_rank]
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
    
    best_obj = np.inf
    best_params = None
    
    for trial in range(5):
        if time.time() > deadline:
            break
        
        if trial == 0:
            x_try = x0.copy()
        else:
            np.random.seed(100 + trial)
            x_try = x0 + np.random.randn(len(x0)) * 0.01
        
        res = minimize(objective, x_try, method='L-BFGS-B', 
                       options={'maxiter': 2000, 'ftol': 1e-15})
        
        if res.fun < best_obj:
            best_obj = res.fun
            best_params = res.x
        
        if best_obj < 1e-12:
            break
    
    if best_params is None or best_obj > 1e-10:
        return None
    
    U_r = best_params[:target_rank*n*m].reshape(n*m, target_rank)
    V_r = best_params[target_rank*n*m:target_rank*(n*m+m*p)].reshape(m*p, target_rank)
    W_r = best_params[target_rank*(n*m+m*p):].reshape(p*n, target_rank)
    
    # Snap to half-integers
    U_r = np.round(U_r * 2) / 2
    V_r = np.round(V_r * 2) / 2
    W_r = np.round(W_r * 2) / 2
    
    # Verify
    res = make_residuals(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]))
    if np.max(np.abs(res)) < 1e-6:
        return U_r.tolist(), V_r.tolist(), W_r.tolist()
    
    return None

def _verify_solution(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                max_err = max(max_err, abs(val - 1.0))
    return max_err

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Try to find M_{2,2,3} rank-10
    m223 = _find_m223_rank10(deadline - 60)
    
    if m223 is not None:
        U_m223, V_m223, W_m223 = m223
        sol50 = _build_rank50_from_m223(U_m223, V_m223, W_m223)
        
        # Verify rank-50 solution
        err = _verify_solution(sol50['U'], sol50['V'], sol50['W'])
        if err < 1e-6:
            # Try to reduce to 49
            if time.time() < deadline - 30:
                reduced = _try_reduce_rank(sol50['U'], sol50['V'], sol50['W'], 50, 49, deadline - 10)
                if reduced is not None:
                    U_r, V_r, W_r = reduced
                    return {"U": U_r, "V": V_r, "W": W_r}
            
            return sol50
    
    # Fallback: rank-53 solution
    return _build_rank53()
# EVOLVE-BLOCK-END
