"""Rank search for 2x2x5 matrix multiplication tensor using FFT-inspired structured circulant embeddings and aggressive ALS."""
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
    R = U_a.shape[1]
    result = np.zeros(T.shape, dtype=np.float64)
    for r in range(R):
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
    
    r = 14
    U[0][r] = 1; V[0*p+4][r] = 1; W[4*n+0][r] = 1
    r = 15
    U[1][r] = 1; V[1*p+4][r] = 1; W[4*n+0][r] = 1
    r = 16
    U[2][r] = 1; V[0*p+4][r] = 1; W[4*n+1][r] = 1
    r = 17
    U[3][r] = 1; V[1*p+4][r] = 1; W[4*n+1][r] = 1
    
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
        sol, res, rank_s, sv = np.linalg.lstsq(VvW_flat, b, rcond=None)
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
        sol, res, rank_s, sv = np.linalg.lstsq(Uw_flat, b, rcond=None)
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
        sol, res, rank_s, sv = np.linalg.lstsq(UV_flat, b, rcond=None)
        W[w] = sol
    return W

def _als_iterate(U, V, W, T_a, n, m, p, R, max_iters=200, tol=1e-14):
    for it in range(max_iters):
        U = _als_solve_U(T_a, V, W, n, m, p, R)
        V = _als_solve_V(T_a, U, W, n, m, p, R)
        W = _als_solve_W(T_a, U, V, n, m, p, R)
        
        err = 0.0
        for r in range(R):
            err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T_a)**2)
        err = np.sqrt(err / (n*m * m*p * p*n))
        
        if it > 0 and abs(prev_err - err) < tol:
            break
        prev_err = err
    return U, V, W

def _try_polish(U, V, W, T, n, m, p, R):
    """Try to round to rational with small denominator after convergence."""
    T_a = np.array(T, dtype=np.float64)
    # Check if already exact
    if _check_decomposition(U, V, W, T):
        for denom in [2, 4, 8, 16, 32, 64]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            if _check_decomposition(U_s, V_s, W_s, T):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _try_fft_circulant_rank(n, m, p, R, T, deadline):
    """
    Try to embed the 2x2x5 multiplication into a cyclic group structure.
    For M_{2,2,5}, we can try rank R = 16 or 17 with structured initializations
    based on discrete Fourier transform patterns.
    """
    N = n * m  # 4
    M = m * p  # 10
    P = p * n  # 10
    
    T_a = np.array(T, dtype=np.float64)
    
    # Try various structured initializations
    inits = []
    
    # FFT-inspired: use roots of unity patterns
    for k in range(4):
        # Create circulant-like patterns with different phase shifts
        phase = 2 * np.pi * k / 8
        U_init = np.zeros((N, R))
        V_init = np.zeros((M, R))
        W_init = np.zeros((P, R))
        
        for r in range(R):
            theta = 2 * np.pi * r / R + phase
            # Structured pattern based on group algebra
            for u in range(N):
                U_init[u, r] = np.cos(theta * (u + 1))
            for v in range(M):
                V_init[v, r] = np.cos(theta * (v + 1) + np.pi/4)
            for w in range(P):
                W_init[w, r] = np.cos(theta * (w + 1) + np.pi/3)
        
        inits.append((U_init, V_init, W_init))
    
    # Rank-18 perturbation: remove 2 random terms and re-solve
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    if R <= 17:
        # Remove terms from rank-18 and perturb
        for seed in range(30):
            if time.time() > deadline:
                break
            np.random.seed(seed * 17 + 42)
            if R == 17:
                remove_idx = np.random.randint(0, 18)
            else:
                remove_idx = np.random.choice(18, size=2, replace=False)
            
            idx = [i for i in range(18) if i not in remove_idx]
            U_init = U18_a[:, idx].copy()
            V_init = V18_a[:, idx].copy()
            W_init = W18_a[:, idx].copy()
            
            # Add small perturbation
            U_init += np.random.randn(N, R) * 0.08
            V_init += np.random.randn(M, R) * 0.08
            W_init += np.random.randn(P, R) * 0.08
            
            inits.append((U_init, V_init, W_init))
    
    # Random Gaussian inits
    for seed in range(40):
        if time.time() > deadline:
            break
        np.random.seed(seed * 23 + 77)
        U_init = np.random.randn(N, R) * 0.6
        V_init = np.random.randn(M, R) * 0.6
        W_init = np.random.randn(P, R) * 0.6
        inits.append((U_init, V_init, W_init))
    
    best_err = np.inf
    best_result = None
    
    for U_init, V_init, W_init in inits:
        if time.time() > deadline:
            break
        
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        U, V, W = _als_iterate(U, V, W, T_a, n, m, p, R, max_iters=150)
        
        # Compute final error
        result = np.zeros(T_a.shape, dtype=np.float64)
        for r in range(R):
            result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
        err = np.linalg.norm(result - T_a) / np.sqrt(T_a.size)
        
        if err < best_err:
            best_err = err
            best_result = (U.copy(), V.copy(), W.copy())
        
        if err < 1e-10:
            polished = _try_polish(U, V, W, T, n, m, p, R)
            if polished is not None:
                return polished
    
    # Try the best result with more iterations
    if best_result is not None and best_err < 1e-5:
        U, V, W = best_result
        U, V, W = _als_iterate(U, V, W, T_a, n, m, p, R, max_iters=300)
        polished = _try_polish(U, V, W, T, n, m, p, R)
        if polished is not None:
            return polished
    
    return None

def _try_rank_specific(n, m, p, R, T, deadline):
    """Try to find rank-R decomposition using FFT-inspired and perturbation strategies."""
    T_a = np.array(T, dtype=np.float64)
    N = n * m
    M = m * p
    P = p * n
    
    # Strategy 1: FFT-inspired structured inits
    result = _try_fft_circulant_rank(n, m, p, R, T, deadline)
    if result is not None:
        return result
    
    # Strategy 2: More aggressive random search with refinement
    for seed in range(50):
        if time.time() > deadline:
            break
        np.random.seed(seed * 31 + 111)
        U_init = np.random.randn(N, R) * 0.5
        V_init = np.random.randn(M, R) * 0.5
        W_init = np.random.randn(P, R) * 0.5
        
        U, V, W = _als_iterate(U_init, V_init, W_init, T_a, n, m, p, R, max_iters=200)
        
        result = np.zeros(T_a.shape, dtype=np.float64)
        for r in range(R):
            result += np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r])
        err = np.linalg.norm(result - T_a) / np.sqrt(T_a.size)
        
        if err < 1e-10:
            polished = _try_polish(U, V, W, T, n, m, p, R)
            if polished is not None:
                return polished
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    T = _build_target()
    
    # Try rank 16
    if time.time() < deadline - 30:
        result16 = _try_rank_specific(n, m, p, 16, T, deadline - 30)
        if result16 is not None:
            return {"U": result16[0], "V": result16[1], "W": result16[2]}
    
    # Try rank 17
    if time.time() < deadline - 15:
        result17 = _try_rank_specific(n, m, p, 17, T, deadline - 15)
        if result17 is not None:
            return {"U": result17[0], "V": result17[1], "W": result17[2]}
    
    # Fallback to rank 18
    U18, V18, W18 = _build_rank18(n, m, p)
    return {"U": U18, "V": V18, "W": W18}
