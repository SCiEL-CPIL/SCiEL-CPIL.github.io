import numpy as np
import time
import sys
from fractions import Fraction

DIMENSIONS = [2, 2, 5]

def _build_target():
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i*m + j
                v = j*p + k
                w = k*n + i
                T[u, v, w] = 1
    return T

def _snap_to_rational(val, max_denom=16):
    for d in range(1, max_denom + 1):
        num = round(val * d)
        if abs(val - num / d) < 1e-9:
            return Fraction(num, d)
    return None

def _check_decomposition_exact_fast(U, V, W, T):
    n, m, p = DIMENSIONS
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    result = np.einsum('ur,vr,wr->uvw', U_a, V_a, W_a)
    if not np.allclose(result, T.astype(np.float64), atol=1e-8, rtol=0):
        return False
    U_frac = np.zeros_like(U_a, dtype=object)
    V_frac = np.zeros_like(V_a, dtype=object)
    W_frac = np.zeros_like(W_a, dtype=object)
    def try_snap_matrix(mat, mat_frac):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                f = _snap_to_rational(mat[i, j], max_denom=8)
                if f is None:
                    return False
                mat_frac[i, j] = f
        return True
    if not try_snap_matrix(U_a, U_frac):
        return False
    if not try_snap_matrix(V_a, V_frac):
        return False
    if not try_snap_matrix(W_a, W_frac):
        return False
    R = U_a.shape[1]
    for u in range(n*m):
        for v in range(m*p):
            for w in range(p*n):
                total = Fraction(0)
                for r in range(R):
                    total += U_frac[u, r] * V_frac[v, r] * W_frac[w, r]
                if total != T[u, v, w]:
                    return False
    return True

def _build_rank18(n, m, p):
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    def add_strassen_block(col0, col1, base):
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1; W[col1*n+1][r] = 1
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[0*p+col0][r] = 1
        W[col0*n+1][r] = 1; W[col1*n+1][r] = -1
        r = base + 2
        U[0][r] = 1
        V[0*p+col1][r] = 1; V[1*p+col1][r] = -1
        W[col1*n+0][r] = 1; W[col1*n+1][r] = 1
        r = base + 3
        U[3][r] = 1
        V[1*p+col0][r] = 1; V[0*p+col0][r] = -1
        W[col0*n+0][r] = 1; W[col0*n+1][r] = 1
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[1*p+col1][r] = 1
        W[col0*n+0][r] = -1; W[col1*n+0][r] = 1
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[0*p+col0][r] = 1; V[0*p+col1][r] = 1
        W[col1*n+1][r] = 1
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[1*p+col0][r] = 1; V[1*p+col1][r] = 1
        W[col0*n+0][r] = 1
    add_strassen_block(0, 1, 0)
    add_strassen_block(2, 3, 7)
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    return U, V, W

def _als_solve_U(T_a, V, W, R):
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    VvW = np.einsum('vr,wr->vwr', V_a, W_a)
    VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
    n, m, p = DIMENSIONS
    U = np.zeros((n*m, R), dtype=np.float64)
    for u in range(n*m):
        b = T_a[u].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(VvW_flat, b, rcond=None)
        U[u] = sol
    return U

def _als_solve_V(T_a, U, W, R):
    U_a = np.array(U, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    Uw = np.einsum('ur,wr->uwr', U_a, W_a)
    Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
    n, m, p = DIMENSIONS
    V = np.zeros((m*p, R), dtype=np.float64)
    for v in range(m*p):
        b = T_a[:, v, :].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(Uw_flat, b, rcond=None)
        V[v] = sol
    return V

def _als_solve_W(T_a, U, V, R):
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    UV = np.einsum('ur,vr->uvr', U_a, V_a)
    UV_flat = UV.reshape(-1, R)
    n, m, p = DIMENSIONS
    W = np.zeros((p*n, R), dtype=np.float64)
    for w in range(p*n):
        b = T_a[:, :, w].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W

def _als_run(U_init, V_init, W_init, T_a, R, max_iters=200, tol=1e-14):
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    prev_err = np.inf
    for it in range(max_iters):
        U = _als_solve_U(T_a, V, W, R)
        V = _als_solve_V(T_a, U, W, R)
        W = _als_solve_W(T_a, U, V, R)
        result = np.einsum('ur,vr,wr->uvw', U, V, W)
        err = np.sqrt(np.mean((result - T_a)**2))
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W, err

def _try_rationalize(U, V, W, T_a):
    for denom in [1, 2, 4, 8, 16]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        result = np.einsum('ur,vr,wr->uvw', U_s, V_s, W_s)
        if np.allclose(result, T_a, atol=1e-10, rtol=0):
            return U_s.tolist(), V_s.tolist(), W_s.tolist()
    return None

def _try_rank17_als(n, m, p, T, deadline):
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    # Strategy 1: Remove each schoolbook term (14-17) and run ALS
    for term_to_remove in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(5):
            if time.time() > deadline:
                break
            np.random.seed(term_to_remove * 100 + seed)
            if seed == 0:
                U_p = U_init.copy()
                V_p = V_init.copy()
                W_p = W_init.copy()
            else:
                scale = 0.05 * (seed)
                U_p = U_init + np.random.randn(4, 17) * scale
                V_p = V_init + np.random.randn(10, 17) * scale
                W_p = W_init + np.random.randn(10, 17) * scale
            
            U_sol, V_sol, W_sol, err = _als_run(U_p, V_p, W_p, T_a, 17, max_iters=100)
            if err < 1e-10:
                rationalized = _try_rationalize(U_sol, V_sol, W_sol, T_a)
                if rationalized is not None:
                    return rationalized
    
    # Strategy 2: Remove each Strassen term and run ALS
    for term_to_remove in range(14):
        if time.time() > deadline:
            break
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(3):
            if time.time() > deadline:
                break
            np.random.seed(1000 + term_to_remove * 100 + seed)
            if seed == 0:
                U_p = U_init.copy()
                V_p = V_init.copy()
                W_p = W_init.copy()
            else:
                scale = 0.1 * (seed)
                U_p = U_init + np.random.randn(4, 17) * scale
                V_p = V_init + np.random.randn(10, 17) * scale
                W_p = W_init + np.random.randn(10, 17) * scale
            
            U_sol, V_sol, W_sol, err = _als_run(U_p, V_p, W_p, T_a, 17, max_iters=80)
            if err < 1e-10:
                rationalized = _try_rationalize(U_sol, V_sol, W_sol, T_a)
                if rationalized is not None:
                    return rationalized
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    T = _build_target()
    
    # Verify rank-18 exists
    U18, V18, W18 = _build_rank18(n, m, p)
    
    # Try to find rank-17 using ALS
    result17 = _try_rank17_als(n, m, p, T, deadline)
    if result17 is not None:
        U, V, W = result17
        if _check_decomposition_exact_fast(U, V, W, T):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to rank-18
    return {
        "U": U18,
        "V": V18,
        "W": W18
    }
