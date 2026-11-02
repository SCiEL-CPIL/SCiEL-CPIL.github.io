"""Combined: rank-18 base with multi-rank search (17, 16, 15) using structured + random strategies."""
import numpy as np
import time
import sys

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

def _check_decomposition(U, V, W, T):
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(U_a.shape[1]):
        result += np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
    return np.allclose(result, T.astype(np.float64), atol=1e-9)

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
    
    r = 14; U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15; U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16; U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17; U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
    return U, V, W

def _als_solve_U(T, V, W, n, m, p, R):
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    VvW = np.einsum('vr,wr->vwr', V_a, W_a)
    VvW_flat = VvW.transpose(0, 2, 1).reshape(-1, R)
    U = np.zeros((n*m, R), dtype=np.float64)
    for u in range(n*m):
        b = T_a[u].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(VvW_flat, b, rcond=None)
        U[u] = sol
    return U

def _als_solve_V(T, U, W, n, m, p, R):
    U_a = np.array(U, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    Uw = np.einsum('ur,wr->uwr', U_a, W_a)
    Uw_flat = Uw.transpose(0, 2, 1).reshape(-1, R)
    V = np.zeros((m*p, R), dtype=np.float64)
    for v in range(m*p):
        b = T_a[:, v, :].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(Uw_flat, b, rcond=None)
        V[v] = sol
    return V

def _als_solve_W(T, U, V, n, m, p, R):
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    UV = np.einsum('ur,vr->uvr', U_a, V_a)
    UV_flat = UV.reshape(-1, R)
    W = np.zeros((p*n, R), dtype=np.float64)
    for w in range(p*n):
        b = T_a[:, :, w].reshape(-1)
        sol, _, _, _ = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W

def _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, R, max_iters=200, tol=1e-13):
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    prev_err = np.inf
    for it in range(max_iters):
        U = _als_solve_U(T_a, V, W, n, m, p, R)
        V = _als_solve_V(T_a, U, W, n, m, p, R)
        W = _als_solve_W(T_a, U, V, n, m, p, R)
        err = 0.0
        for r in range(R):
            err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T_a)**2)
        err = np.sqrt(err / (n*m * m*p * p*n))
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    if _check_decomposition(U, V, W, T_a):
        for denom in [2, 4, 8, 16, 32, 64]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            if _check_decomposition(U_s, V_s, W_s, T_a):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _structured_init(n, m, p, R, seed):
    """Create structured initializations exploiting tensor symmetries."""
    np.random.seed(seed)
    U = np.zeros((n*m, R), dtype=np.float64)
    V = np.zeros((m*p, R), dtype=np.float64)
    W = np.zeros((p*n, R), dtype=np.float64)
    
    # Use Kronecker-like structure
    # For 2x2x5, exploit the 2x2 block structure
    for r in range(R):
        # Random but structured signs
        signs_u = np.random.choice([-1, 1], size=n*m)
        signs_v = np.random.choice([-1, 1], size=m*p)
        signs_w = np.random.choice([-1, 1], size=p*n)
        
        U[:, r] = signs_u * (0.5 + 0.5 * np.random.rand(n*m))
        V[:, r] = signs_v * (0.5 + 0.5 * np.random.rand(m*p))
        W[:, r] = signs_w * (0.5 + 0.5 * np.random.rand(p*n))
    
    return U, V, W

def _try_rank_search(n, m, p, T, R, deadline, max_tries=30):
    """Try to find a rank-R decomposition using multiple strategies."""
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    tries = 0
    
    # Strategy 1: Remove terms from rank-18 and perturb
    if R == 17:
        for term_to_remove in range(18):
            if time.time() > deadline or tries >= max_tries:
                break
            idx = [i for i in range(18) if i != term_to_remove]
            U_init = U18_a[:, idx].copy()
            V_init = V18_a[:, idx].copy()
            W_init = W18_a[:, idx].copy()
            
            for seed in range(3):
                if time.time() > deadline or tries >= max_tries:
                    break
                np.random.seed(term_to_remove * 100 + seed)
                scale = 0.005 + 0.003 * (seed % 5)
                U_pert = U_init + np.random.randn(4, 17) * scale
                V_pert = V_init + np.random.randn(10, 17) * scale
                W_pert = W_init + np.random.randn(10, 17) * scale
                result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 17, max_iters=200)
                tries += 1
                if result is not None:
                    return result
    
    # Strategy 2: Structured random initializations
    for seed in range(max_tries):
        if time.time() > deadline:
            break
        U_init, V_init, W_init = _structured_init(n, m, p, R, seed * 31 + 7)
        result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, R, max_iters=150)
        if result is not None:
            return result
    
    # Strategy 3: Pure random with varied scales
    for seed in range(max_tries):
        if time.time() > deadline:
            break
        np.random.seed(seed * 13 + 42)
        scale = 0.05 + 0.1 * (seed % 10)
        U_init = np.random.randn(n*m, R) * scale
        V_init = np.random.randn(m*p, R) * scale
        W_init = np.random.randn(p*n, R) * scale
        result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, R, max_iters=100)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s kill
    T = _build_target()
    U18, V18, W18 = _build_rank18(n, m, p)
    
    if not _check_decomposition(U18, V18, W18, T):
        return {"U": U18, "V": V18, "W": W18}
    
    # Try rank-17 first (most likely to succeed)
    result17 = _try_rank_search(n, m, p, T, 17, deadline, max_tries=40)
    if result17 is not None:
        U, V, W = result17
        if _check_decomposition(U, V, W, T):
            return {"U": U, "V": V, "W": W}
    
    # If time remains, try rank-16
    if time.time() < deadline - 10:
        result16 = _try_rank_search(n, m, p, T, 16, deadline, max_tries=25)
        if result16 is not None:
            U, V, W = result16
            if _check_decomposition(U, V, W, T):
                return {"U": U, "V": V, "W": W}
    
    # If time remains, try rank-15
    if time.time() < deadline - 5:
        result15 = _try_rank_search(n, m, p, T, 15, deadline, max_tries=15)
        if result15 is not None:
            U, V, W = result15
            if _check_decomposition(U, V, W, T):
                return {"U": U, "V": V, "W": W}
    
    return {"U": U18, "V": V18, "W": W18}
