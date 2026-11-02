"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, least_squares

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

def _residuals_m223(params):
    """Compute residuals for M_{2,2,3} decomposition."""
    n, m, p = 2, 2, 3
    rank = 10
    U = params[:n*m*rank].reshape(n*m, rank)
    V = params[n*m*rank:(n*m*rank + m*p*rank)].reshape(m*p, rank)
    W = params[(n*m*rank + m*p*rank):].reshape(p*n, rank)
    
    res = np.zeros(n*m*p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                res[idx] = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i]) - (1.0 if i == k else 0.0)
                idx += 1
    return res

def _verify_m223(U, V, W, tol=1e-9):
    """Verify if U, V, W form a valid M_{2,2,3} decomposition."""
    n, m, p = 2, 2, 3
    max_res = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                expected = 1.0 if i == k else 0.0
                max_res = max(max_res, abs(val - expected))
    return max_res < tol

def _find_decomp_m223_rank10(deadline, max_seeds=50):
    """Search for rank-10 decomposition of M_{2,2,3}."""
    n, m, p = 2, 2, 3
    rank = 10
    total_vars = (n*m + m*p + p*n) * rank
    
    def objective(params):
        res = _residuals_m223(params)
        return np.sum(res**2)
    
    best_params = None
    best_obj = np.inf
    
    for seed in range(max_seeds):
        if time.time() > deadline:
            break
        
        np.random.seed(seed)
        
        # Initialize with small random values
        # Sometimes it helps to start with values that are closer to the expected magnitude
        scale = 0.5
        x0 = np.random.randn(total_vars) * scale
        
        # Add some structure: ensure some entries are 0 or 1 to help snapping
        if seed % 2 == 0:
            mask = np.random.rand(total_vars) < 0.2
            x0[mask] = np.random.choice([-1, 0, 1], size=mask.sum())
            
        try:
            # First pass: L-BFGS-B
            res = minimize(objective, x0, method='L-BFGS-B', 
                          options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.fun < best_obj:
                best_obj = res.fun
                best_params = res.x
                
            if best_obj < 1e-10:
                break
                
        except Exception:
            continue
    
    if best_params is None or best_obj > 1e-8:
        return None
    
    U = best_params[:n*m*rank].reshape(n*m, rank)
    V = best_params[n*m*rank:(n*m*rank + m*p*rank)].reshape(m*p, rank)
    W = best_params[(n*m*rank + m*p*rank):].reshape(p*n, rank)
    
    # Polish with least_squares for higher precision
    try:
        res_ls = least_squares(_residuals_m223, best_params, method='trf', 
                               ftol=1e-15, xtol=1e-15, gtol=1e-15, max_nfev=2000)
        if res_ls.cost < best_obj:
            best_params = res_ls.x
            best_obj = res_ls.cost
            U = best_params[:n*m*rank].reshape(n*m, rank)
            V = best_params[n*m*rank:(n*m*rank + m*p*rank)].reshape(m*p, rank)
            W = best_params[(n*m*rank + m*p*rank):].reshape(p*n, rank)
    except Exception:
        pass
    
    # Try snapping to half-integers
    for mult in [2, 4, 8]:
        U_snap = np.round(U * mult) / mult
        V_snap = np.round(V * mult) / mult
        W_snap = np.round(W * mult) / mult
        
        if _verify_m223(U_snap, V_snap, W_snap, tol=1e-6):
            return U_snap, V_snap, W_snap
    
    # If no exact snap works, return the float solution if residual is tiny
    if best_obj < 1e-12:
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
    
    # Schoolbook for 2x2x1
    U_school = np.eye(4)
    V_school = np.array([[1, 0, 1, 0],
                         [0, 1, 0, 1]], dtype=float)
    W_school = np.array([[1, 1, 0, 0],
                         [0, 0, 1, 1]], dtype=float)
    
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
    
    U3 = np.asarray(U3, dtype=float)
    V3 = np.asarray(V3, dtype=float)
    W3 = np.asarray(W3, dtype=float)
    
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
    # This would give us a rank-50 solution (5 blocks of rank 10)
    m223 = _find_decomp_m223_rank10(deadline, max_seeds=50)
    
    if m223 is not None:
        U3, V3, W3 = m223
        result = _build_rank50(U3, V3, W3)
        # Verify the result
        U_arr = np.array(result['U'], dtype=float)
        V_arr = np.array(result['V'], dtype=float)
        W_arr = np.array(result['W'], dtype=float)
        
        # Check that the decomposition is valid
        max_res = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    expected = 1.0 if i == k else 0.0
                    max_res = max(max_res, abs(val - expected))
        
        if max_res < 1e-6:
            return result
    
    # Fallback to rank-53
    return fallback
# EVOLVE-BLOCK-END
