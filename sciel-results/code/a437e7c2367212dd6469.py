import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def _try_find_low_rank_333(timeout=10):
    """
    Attempts to find a rank 23 decomposition for 3x3x3 matrix multiplication.
    Returns (U, V, W) or None if not found.
    """
    n, m, p = 3, 3, 3
    R = 23
    
    start = time.time()
    
    # Target tensor T[i, j, k] = 1
    # We want sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    np.random.seed(42)
    U = np.random.randn(9, R)
    V = np.random.randn(9, R)
    W = np.random.randn(9, R)
    
    def residuals(params):
        U_flat = params[:9*R].reshape(9, R)
        V_flat = params[9*R:18*R].reshape(9, R)
        W_flat = params[18*R:27*R].reshape(9, R)
        
        res = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    u_idx = i*3 + j
                    v_idx = j*3 + k
                    w_idx = k*3 + i
                    val = np.dot(U_flat[u_idx, :], V_flat[v_idx, :] * W_flat[w_idx, :])
                    res.append(val - 1.0)
        return np.array(res)
    
    from scipy.optimize import least_squares
    
    x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        # Use a timeout wrapper? Least squares doesn't have a direct timeout.
        # We can just run it and hope it's fast.
        # If it takes too long, we might kill the whole program.
        # To be safe, we limit the max iterations.
        result = least_squares(residuals, x0, method='trf', xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=1000)
        
        if time.time() - start > timeout:
            return None
            
        if result.success and np.max(np.abs(result.fun)) < 1e-9:
            U_sol = result.x[:9*R].reshape(9, R)
            V_sol = result.x[9*R:18*R].reshape(9, R)
            W_sol = result.x[18*R:27*R].reshape(9, R)
            
            U_round = np.round(U_sol * 2) / 2
            V_round = np.round(V_sol * 2) / 2
            W_round = np.round(W_sol * 2) / 2
            
            exact = True
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        u_idx = i*3 + j
                        v_idx = j*3 + k
                        w_idx = k*3 + i
                        val = np.dot(U_round[u_idx, :], V_round[v_idx, :] * W_round[w_idx, :])
                        if abs(val - 1.0) > 1e-9:
                            exact = False
                            break
                    if not exact: break
                if not exact: break
            
            if exact:
                return U_round.tolist(), V_round.tolist(), W_round.tolist()
                
    except Exception:
        pass
        
    return None

def construct():
    n, m, p = DIMENSIONS  # n=3, m=3, p=7
    start_time = time.time()
    
    # Try to find low rank for 3x3x3 part
    U_333, V_333, W_333 = _try_find_low_rank_333(timeout=10)
    
    if U_333 is not None:
        p1 = 3
        p2 = 4
        r1 = 23
        r2 = 3 * 3 * 4 # 36
        total_rank = r1 + r2
        
        # U_333 is 9x23 list of lists
        U = [row[:] + [0]*r2 for row in U_333]
        V = [row[:] + [0]*r2 for row in V_333]
        W = [row[:] + [0]*r2 for row in W_333]
        
        # Add schoolbook for 3x3x4
        for i in range(n):
            for j in range(m):
                for k_rel in range(p2):
                    k_abs = p1 + k_rel
                    col_idx = r1 + (i*m + j)*p2 + k_rel
                    U[i * m + j][col_idx] = 1
                    V[j * p + k_abs][col_idx] = 1
                    W[k_abs * n + i][col_idx] = 1
                    
        return {"U": U, "V": V, "W": W}
    else:
        total_rank = n * m * p
        U = [[0] * total_rank for _ in range(n * m)]
        V = [[0] * total_rank for _ in range(m * p)]
        W = [[0] * total_rank for _ in range(p * n)]
        
        col_idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col_idx] = 1
                    V[j * p + k][col_idx] = 1
                    W[k * n + i][col_idx] = 1
                    col_idx += 1
                    
        return {"U": U, "V": V, "W": W}
