"""
Enhanced search for rank-17 decomposition of 2x2 @ 2x5 multiplication tensor.
Key improvements: faster ALS (precomputed products), more diverse initializations,
structured perturbation, and rational polishing.
"""
import numpy as np
import time
import sys
from itertools import product

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
    """Build the known rank-18 Strassen+schoolbook decomposition."""
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

def _fast_als(T_a, U, V, W, n, m, p, R, max_iters=300, tol=1e-14):
    """
    Fast ALS with precomputed products. U is (nm,R), V is (mp,R), W is (pn,R).
    """
    nm, mp, pn = n*m, m*p, p*n
    
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    prev_err = np.inf
    for it in range(max_iters):
        # Solve for U: for each u, T[u] = sum_r U[u,r] * (V*W)[v,w,r]
        # Precompute VW[r] = outer(V[:,r], W[:,r]) shape (mp, pn)
        # Then T[u] (shape mp*pn) = VW_flat (mp*pn, R) @ U[u]
        VW = np.einsum('vr,wr->rvp', V, W)  # (R, mp, pn)
        VW_flat = VW.reshape(R, -1).T  # (mp*pn, R)
        
        # U[u] = argmin ||T[u] - VW_flat @ U[u]||
        U_new = np.zeros((nm, R), dtype=np.float64)
        for u in range(nm):
            b = T_a[u].reshape(-1)
            sol = np.linalg.lstsq(VW_flat, b, rcond=None)[0]
            U_new[u] = sol
        U = U_new
        
        # Solve for V: for each v, T[:,v,:] = sum_r V[v,r] * (U*W)[u,w,r]
        UW = np.einsum('ur,wr->rwp', U, W)  # (R, nm, pn)
        UW_flat = UW.reshape(R, -1).T  # (nm*pn, R)
        
        V_new = np.zeros((mp, R), dtype=np.float64)
        for v in range(mp):
            b = T_a[:, v, :].reshape(-1)
            sol = np.linalg.lstsq(UW_flat, b, rcond=None)[0]
            V_new[v] = sol
        V = V_new
        
        # Solve for W: for each w, T[:,:,w] = sum_r W[w,r] * (U*V)[u,v,r]
        UV = np.einsum('ur,vr->rvp', U, V)  # (R, nm, mp)
        UV_flat = UV.reshape(R, -1).T  # (nm*mp, R)
        
        W_new = np.zeros((pn, R), dtype=np.float64)
        for w in range(pn):
            b = T_a[:, :, w].reshape(-1)
            sol = np.linalg.lstsq(UV_flat, b, rcond=None)[0]
            W_new[w] = sol
        W = W_new
        
        # Compute error
        err = 0.0
        for r in range(R):
            diff = np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T_a
            err += np.sum(diff * diff)
        err = np.sqrt(err / (nm * mp * pn))
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    return U, V, W, err

def _polish_rational(U, V, W, T_a, R):
    """Try to find rational/half-integer representation."""
    for denom in [2, 4, 8, 16, 32, 64, 128, 256]:
        U_s = np.round(U * denom) / denom
        V_s = np.round(V * denom) / denom
        W_s = np.round(W * denom) / denom
        if _check_decomposition(U_s, V_s, W_s, T_a.astype(int)):
            return U_s.tolist(), V_s.tolist(), W_s.tolist()
    # Return as-is if close
    if _check_decomposition(U, V, W, T_a.astype(int)):
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _try_rank17(n, m, p, T, deadline):
    """Search for rank-17 decomposition with multiple strategies."""
    T_a = T.astype(np.float64)
    nm, mp, pn = n*m, m*p, p*n
    R = 17
    
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    best_result = None
    
    # Strategy 1: Remove each of the 4 schoolbook terms, perturb, ALS
    for term_to_remove in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(20):
            if time.time() > deadline:
                break
            np.random.seed(1000 + term_to_remove * 100 + seed)
            scale = 0.01 + 0.01 * (seed % 5)
            U_pert = U_init + np.random.randn(nm, R) * scale
            V_pert = V_init + np.random.randn(mp, R) * scale
            W_pert = W_init + np.random.randn(pn, R) * scale
            
            U, V, W, err = _fast_als(T_a, U_pert, V_pert, W_pert, n, m, p, R, max_iters=400, tol=1e-14)
            if err < 1e-10:
                result = _polish_rational(U, V, W, T_a, R)
                if result is not None:
                    return result
                return U.tolist(), V.tolist(), W.tolist()
    
    # Strategy 2: Remove Strassen terms and perturb
    for term_to_remove in range(14):
        if time.time() > deadline:
            break
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(10):
            if time.time() > deadline:
                break
            np.random.seed(2000 + term_to_remove * 100 + seed)
            scale = 0.01 + 0.01 * (seed % 5)
            U_pert = U_init + np.random.randn(nm, R) * scale
            V_pert = V_init + np.random.randn(mp, R) * scale
            W_pert = W_init + np.random.randn(pn, R) * scale
            
            U, V, W, err = _fast_als(T_a, U_pert, V_pert, W_pert, n, m, p, R, max_iters=350, tol=1e-14)
            if err < 1e-10:
                result = _polish_rational(U, V, W, T_a, R)
                if result is not None:
                    return result
                return U.tolist(), V.tolist(), W.tolist()
    
    # Strategy 3: Random initializations with varying scales
    for seed in range(50):
        if time.time() > deadline:
            break
        np.random.seed(3000 + seed * 7)
        scale = 0.1 + 0.2 * (seed % 5)
        U_init = np.random.randn(nm, R) * scale
        V_init = np.random.randn(mp, R) * scale
        W_init = np.random.randn(pn, R) * scale
        
        U, V, W, err = _fast_als(T_a, U_init, V_init, W_init, n, m, p, R, max_iters=200, tol=1e-13)
        if err < 1e-10:
            result = _polish_rational(U, V, W, T_a, R)
            if result is not None:
                return result
            return U.tolist(), V.tolist(), W.tolist()
    
    # Strategy 4: Structured perturbation - mix two removed terms
    for i in range(14):
        if time.time() > deadline:
            break
        for j in range(i+1, 14):
            if time.time() > deadline:
                break
            idx = [k for k in range(18) if k != i and k != j]
            U_init = U18_a[:, idx].copy()
            V_init = V18_a[:, idx].copy()
            W_init = W18_a[:, idx].copy()
            
            np.random.seed(4000 + i * 100 + j)
            new_U = np.random.randn(nm, 1) * 0.3
            new_V = np.random.randn(mp, 1) * 0.3
            new_W = np.random.randn(pn, 1) * 0.3
            
            U_full = np.hstack([U_init, new_U])
            V_full = np.hstack([V_init, new_V])
            W_full = np.hstack([W_init, new_W])
            
            U, V, W, err = _fast_als(T_a, U_full, V_full, W_full, n, m, p, R, max_iters=200, tol=1e-13)
            if err < 1e-10:
                result = _polish_rational(U, V, W, T_a, R)
                if result is not None:
                    return result
                return U.tolist(), V.tolist(), W.tolist()
    
    # Strategy 5: SVD-informed initialization
    try:
        # Flatten T as (nm, mp*pn)
        T_flat = T_a.reshape(nm, mp * pn)
        U_svd, S, Vt = np.linalg.svd(T_flat, full_matrices=False)
        
        for seed in range(20):
            if time.time() > deadline:
                break
            np.random.seed(5000 + seed * 11)
            # Mix SVD structure with random
            if U_svd.shape[1] >= R:
                U_init = 0.3 * U_svd[:, :R] + 0.7 * np.random.randn(nm, R) * 0.2
            else:
                U_init = np.random.randn(nm, R) * 0.2
            V_init = np.random.randn(mp, R) * 0.2
            W_init = np.random.randn(pn, R) * 0.2
            
            U, V, W, err = _fast_als(T_a, U_init, V_init, W_init, n, m, p, R, max_iters=200, tol=1e-13)
            if err < 1e-10:
                result = _polish_rational(U, V, W, T_a, R)
                if result is not None:
                    return result
                return U.tolist(), V.tolist(), W.tolist()
    except:
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    T = _build_target()
    
    # Build and verify rank-18
    U18, V18, W18 = _build_rank18(n, m, p)
    
    if _check_decomposition(U18, V18, W18, T):
        # Try rank-17
        result17 = _try_rank17(n, m, p, T, deadline)
        if result17 is not None:
            U, V, W = result17
            if _check_decomposition(U, V, W, T):
                return {"U": U, "V": V, "W": W}
    
    # Fallback: rank-18
    return {"U": U18, "V": V18, "W": W18}
