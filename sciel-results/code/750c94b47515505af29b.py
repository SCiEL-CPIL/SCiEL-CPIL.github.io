"""Tensor decomposition for 3x3x8 matrix multiplication using ALS refinement."""
import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def _build_schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    T = np.zeros((n, m, p), dtype=float)
    for r in range(rank):
        Ur = Ua[:, r]
        Vr = Va[:, r]
        Wr = Wa[:, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] += Ur[i*m+j] * Vr[j*p+k] * Wr[k*n+i]
    return np.allclose(T, 1.0, atol=1e-6)

def _als_solve_W(U, V, r):
    """Solve for W given U and V using vectorized least squares."""
    n, m, p = DIMENSIONS
    W = np.zeros((p * n, r))
    for k in range(p):
        for i in range(n):
            # A[j, s] = U[i*m+j, s] * V[j*p+k, s]
            A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
            # Solve A @ w = ones(m)
            try:
                w, _, _, _ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                W[k*n+i, :] = w
            except:
                W[k*n+i, :] = 0
    return W

def _als_solve_U(U, V, W, r):
    """Solve for U given V and W."""
    n, m, p = DIMENSIONS
    U_new = np.zeros((n * m, r))
    for i in range(n):
        for j in range(m):
            # A[k, s] = V[j*p+k, s] * W[k*n+i, s]
            A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
            try:
                u, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U_new[i*m+j, :] = u
            except:
                U_new[i*m+j, :] = 0
    return U_new

def _als_solve_V(U, V, W, r):
    """Solve for V given U and W."""
    n, m, p = DIMENSIONS
    V_new = np.zeros((m * p, r))
    for j in range(m):
        for k in range(p):
            # A[i, s] = U[i*m+j, s] * W[k*n+i, s]
            A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
            try:
                v, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                V_new[j*p+k, :] = v
            except:
                V_new[j*p+k, :] = 0
    return V_new

def _compute_loss(U, V, W):
    n, m, p = DIMENSIONS
    total = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                total += (val - 1.0) ** 2
    return total

def _try_rank_als(r, max_sweeps=100, timeout=120.0, start_time=0):
    """Try to find a valid decomposition of rank r using ALS."""
    n, m, p = DIMENSIONS
    
    np.random.seed(42 + r)
    U = np.random.randn(n * m, r)
    V = np.random.randn(m * p, r)
    W = np.random.randn(p * n, r)
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for sweep in range(max_sweeps):
        if time.time() - start_time > timeout - 5:
            break
        
        W = _als_solve_W(U, V, r)
        U = _als_solve_U(U, V, W, r)
        V = _als_solve_V(U, V, W, r)
        
        loss = _compute_loss(U, V, W)
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            break
    
    if best_loss < 1e-8:
        # Polish: try to round to half-integers
        U_r = np.round(best_U * 2) / 2
        V_r = np.round(best_V * 2) / 2
        W_r = np.round(best_W * 2) / 2
        
        max_err = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i])
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-6:
            return {
                "U": [[float(x) for x in row] for row in U_r],
                "V": [[float(x) for x in row] for row in V_r],
                "W": [[float(x) for x in row] for row in W_r]
            }
    
    return None

def construct():
    start_time = time.time()
    timeout = 120.0
    n, m, p = DIMENSIONS
    
    schoolbook = _build_schoolbook()
    
    # Try ranks from 71 down to 50 with time budgeting
    for r in range(71, 49, -1):
        if time.time() - start_time > timeout - 10:
            break
        
        result = _try_rank_als(r, max_sweeps=50, timeout=timeout, start_time=start_time)
        if result is not None:
            return result
    
    return schoolbook
