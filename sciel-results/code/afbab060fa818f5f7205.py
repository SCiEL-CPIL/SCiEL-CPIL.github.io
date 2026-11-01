import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def _add_strassen_block(U, V, W, col, i_off, j_off, k_off):
    """Add Strassen 2x2x2 block at given offsets. Returns new col."""
    n, m, p = 2, 3, 5
    U[i_off * m + j_off][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[j_off * p + k_off][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    U[(i_off+1) * m + j_off][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[j_off * p + k_off][col] = 1
    W[k_off * n + (i_off+1)][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = -1
    col += 1
    
    U[i_off * m + j_off][col] = 1
    V[j_off * p + (k_off+1)][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = -1
    W[(k_off+1) * n + i_off][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    U[(i_off+1) * m + (j_off+1)][col] = 1
    V[(j_off+1) * p + k_off][col] = 1
    V[j_off * p + k_off][col] = -1
    W[k_off * n + i_off][col] = 1
    W[k_off * n + (i_off+1)][col] = 1
    col += 1
    
    U[i_off * m + j_off][col] = 1
    U[i_off * m + (j_off+1)][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = -1
    W[(k_off+1) * n + i_off][col] = 1
    col += 1
    
    U[(i_off+1) * m + j_off][col] = 1
    U[i_off * m + j_off][col] = -1
    V[j_off * p + k_off][col] = 1
    V[j_off * p + (k_off+1)][col] = 1
    W[(k_off+1) * n + (i_off+1)][col] = 1
    col += 1
    
    U[i_off * m + (j_off+1)][col] = 1
    U[(i_off+1) * m + (j_off+1)][col] = -1
    V[(j_off+1) * p + k_off][col] = 1
    V[(j_off+1) * p + (k_off+1)][col] = 1
    W[k_off * n + i_off][col] = 1
    col += 1
    
    return col

def _verify_223(U, V, W):
    """Verify rank-10 decomposition of M_{2,2,3}."""
    for i in range(2):
        for j in range(2):
            for k in range(3):
                s = sum(U[r][i][j] * V[r][j][k] * W[r][k][i] for r in range(10))
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _try_223_rank10(max_time=12):
    """Search for rank-10 decomposition of M_{2,2,3} with half-integer entries."""
    start = time.time()
    from scipy.optimize import minimize
    
    def objective(x):
        U = x[0:40].reshape(10, 2, 2)
        V = x[40:100].reshape(10, 2, 3)
        W = x[100:160].reshape(10, 3, 2)
        err = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    s = float(np.dot(U[:, i, j], V[:, j, k] * W[:, k, i]))
                    err += (s - 1.0) ** 2
        return err
    
    best_val = np.inf
    best_x = None
    
    for restart in range(40):
        if time.time() - start > max_time - 3:
            break
        np.random.seed(1000 + restart)
        x0 = np.random.randn(160) * 0.5
        try:
            res = minimize(objective, x0, method='L-BFGS-B',
                           options={'maxiter': 400, 'ftol': 1e-16, 'gtol': 1e-16})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
            if best_val < 1e-12:
                break
        except Exception:
            continue
    
    if best_x is None or best_val > 1e-10:
        return None
    
    U = best_x[0:40].reshape(10, 2, 2)
    V = best_x[40:100].reshape(10, 2, 3)
    W = best_x[100:160].reshape(10, 3, 2)
    
    # Try rounding to half-integers
    for mult in [1, 2, 3, 4, 5, 6]:
        U_r = np.round(U * mult) / mult
        V_r = np.round(V * mult) / mult
        W_r = np.round(W * mult) / mult
        if _verify_223(U_r.tolist(), V_r.tolist(), W_r.tolist()):
            return (U_r.tolist(), V_r.tolist(), W_r.tolist())
    
    return None

def construct():
    n, m, p = 2, 3, 5
    deadline = time.time() + 140
    
    # Try to find rank-10 for 2x2x3 to achieve total rank 27
    dec223 = None
    if time.time() < deadline - 15:
        t_left = min(12, deadline - time.time() - 5)
        dec223 = _try_223_rank10(max_time=t_left)
    
    if dec223 is not None:
        U223, V223, W223 = dec223
        rank = 27
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        # Block 1: j=0, all k=0..4 -> 10 terms
        for i in range(n):
            for k in range(p):
                U[i * m + 0][col] = 1
                V[0 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        
        # Block 2a: j={1,2}, k={0,1,2} -> rank-10 decomposition
        i_off, j_off, k_off = 0, 1, 0
        for r in range(10):
            for i in range(2):
                for j in range(2):
                    U[(i_off + i) * m + (j_off + j)][col] += U223[r][i][j]
            for j in range(2):
                for k in range(3):
                    V[(j_off + j) * p + (k_off + k)][col] += V223[r][j][k]
            for k in range(3):
                for i in range(2):
                    W[(k_off + k) * n + (i_off + i)][col] += W223[r][k][i]
            col += 1
        
        # Block 2b: j={1,2}, k={3,4} -> Strassen 2x2x2 -> 7 terms
        col = _add_strassen_block(U, V, W, col, 0, 1, 3)
        
        assert col == rank, f"Column mismatch: {col} != {rank}"
        return {"U": U, "V": V, "W": W}
    
    # Fallback: rank-28 solution
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    col = _add_strassen_block(U, V, W, col, 0, 1, 0)
    
    for i in range(n):
        for j in range(1, 3):
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    
    col = _add_strassen_block(U, V, W, col, 0, 1, 3)
    
    assert col == rank, f"Column mismatch: {col} != {rank}"
    return {"U": U, "V": V, "W": W}
