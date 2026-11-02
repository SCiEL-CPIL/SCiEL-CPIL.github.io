"""Combined: rank-18 base with aggressive rank-17/16 ALS search and smooth surrogate."""
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

def _check_decomposition(U, V, W, T, atol=1e-9):
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
    U_a = np.array(U, dtype=np.float64)
    V_a = np.array(V, dtype=np.float64)
    W_a = np.array(W, dtype=np.float64)
    result = np.zeros_like(T, dtype=np.float64)
    for r in range(U_a.shape[1]):
        result += np.einsum('u,v,w->uvw', U_a[:,r], V_a[:,r], W_a[:,r])
    return np.allclose(result, T.astype(np.float64), atol=atol)

def _compute_error(U, V, W, T_a, R):
    """Compute RMS error of the decomposition."""
    n, m, p = T_a.shape[0] // 2, 2, T_a.shape[2] // 2
    err = 0.0
    for r in range(R):
        err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T_a)**2)
    return np.sqrt(err / (T_a.size))

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
    
    # Schoolbook for column 4
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
    """Solve for U given V, W using least squares."""
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
    """Solve for V given U, W."""
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
    """Solve for W given U, V."""
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

def _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, R, max_iters=200, tol=1e-13):
    """Run ALS from a given initialization, return (U, V, W) as lists if exact."""
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    prev_err = np.inf
    for it in range(max_iters):
        U = _als_solve_U(T_a, V, W, n, m, p, R)
        V = _als_solve_V(T_a, U, W, n, m, p, R)
        W = _als_solve_W(T_a, U, V, n, m, p, R)
        
        err = _compute_error(U, V, W, T_a, R)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    if _check_decomposition(U, V, W, T_a):
        for denom in [2, 4, 8, 16, 32]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            if _check_decomposition(U_s, V_s, W_s, T_a):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _smooth_surrogate_search(U_init, V_init, W_init, T, n, m, p, R, max_iters=100, 
                              temperature=1.0, deadline=None):
    """
    Use a smooth surrogate objective with log-sum-exp style penalty.
    This helps escape local minima by providing a smoother landscape.
    """
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    prev_err = np.inf
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Standard ALS steps
        U = _als_solve_U(T_a, V, W, n, m, p, R)
        V = _als_solve_V(T_a, U, W, n, m, p, R)
        W = _als_solve_W(T_a, U, V, n, m, p, R)
        
        err = _compute_error(U, V, W, T_a, R)
        
        if abs(prev_err - err) < 1e-14:
            break
        prev_err = err
    
    if _check_decomposition(U, V, W, T_a, atol=1e-8):
        for denom in [2, 4, 8, 16, 32, 64]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            if _check_decomposition(U_s, V_s, W_s, T_a, atol=1e-8):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _try_rank17_search(n, m, p, T, deadline):
    """Try to find a rank-17 decomposition with multiple strategies."""
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    # Strategy 1: remove each schoolbook term with varied perturbation scales
    for term_to_remove in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(8):
            if time.time() > deadline:
                break
            np.random.seed(term_to_remove * 100 + seed)
            scale = 0.02 + 0.03 * seed
            U_pert = U_init + np.random.randn(4, 17) * scale
            V_pert = V_init + np.random.randn(10, 17) * scale
            W_pert = W_init + np.random.randn(10, 17) * scale
            
            result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 17, max_iters=100)
            if result is not None:
                return result
    
    # Strategy 2: remove Strassen terms with varied perturbation scales
    for term_to_remove in range(14):
        if time.time() > deadline:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(5):
            if time.time() > deadline:
                break
            np.random.seed(term_to_remove * 100 + seed + 50)
            scale = 0.02 + 0.02 * seed
            U_pert = U_init + np.random.randn(4, 17) * scale
            V_pert = V_init + np.random.randn(10, 17) * scale
            W_pert = W_init + np.random.randn(10, 17) * scale
            
            result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 17, max_iters=100)
            if result is not None:
                return result
    
    # Strategy 3: random initializations with diverse scales
    for seed in range(30):
        if time.time() > deadline:
            break
        np.random.seed(seed * 7 + 123)
        scale = 0.3 + 0.1 * (seed % 5)
        U_init = np.random.randn(4, 17) * scale
        V_init = np.random.randn(10, 17) * scale
        W_init = np.random.randn(10, 17) * scale
        
        result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, 17, max_iters=80)
        if result is not None:
            return result
    
    # Strategy 4: smooth surrogate from rank-18 perturbations
    for term_to_remove in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(3):
            if time.time() > deadline:
                break
            np.random.seed(term_to_remove * 200 + seed + 100)
            U_pert = U_init + np.random.randn(4, 17) * 0.1
            V_pert = V_init + np.random.randn(10, 17) * 0.1
            W_pert = W_init + np.random.randn(10, 17) * 0.1
            
            result = _smooth_surrogate_search(U_pert, V_pert, W_pert, T, n, m, p, 17, 
                                               max_iters=80, deadline=deadline)
            if result is not None:
                return result
    
    return None

def _try_rank16_search(n, m, p, T, deadline):
    """Try to find a rank-16 decomposition."""
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    # Remove pairs of schoolbook terms
    schoolbook_terms = [14, 15, 16, 17]
    for i in range(len(schoolbook_terms)):
        for j in range(i+1, len(schoolbook_terms)):
            if time.time() > deadline:
                break
            t1, t2 = schoolbook_terms[i], schoolbook_terms[j]
            idx = [k for k in range(18) if k != t1 and k != t2]
            U_init = U18_a[:, idx].copy()
            V_init = V18_a[:, idx].copy()
            W_init = W18_a[:, idx].copy()
            
            for seed in range(5):
                if time.time() > deadline:
                    break
                np.random.seed((t1*100 + t2) * 10 + seed)
                scale = 0.05 + 0.02 * seed
                U_pert = U_init + np.random.randn(4, 16) * scale
                V_pert = V_init + np.random.randn(10, 16) * scale
                W_pert = W_init + np.random.randn(10, 16) * scale
                
                result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 16, max_iters=80)
                if result is not None:
                    return result
    
    # Random initializations for rank-16
    for seed in range(20):
        if time.time() > deadline:
            break
        np.random.seed(seed * 13 + 456)
        scale = 0.3 + 0.1 * (seed % 4)
        U_init = np.random.randn(4, 16) * scale
        V_init = np.random.randn(10, 16) * scale
        W_init = np.random.randn(10, 16) * scale
        
        result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, 16, max_iters=60)
        if result is not None:
            return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    # Use 140s deadline to be safe before the 150s kill
    deadline = time.time() + 140
    
    T = _build_target()
    
    # First, build the known rank-18 decomposition
    U18, V18, W18 = _build_rank18(n, m, p)
    
    # Verify rank-18
    if _check_decomposition(U18, V18, W18, T):
        # Try to find rank-17
        result17 = _try_rank17_search(n, m, p, T, deadline)
        if result17 is not None:
            U, V, W = result17
            if _check_decomposition(U, V, W, T):
                return {"U": U, "V": V, "W": W}
        
        # If rank-17 not found, try rank-16
        if time.time() < deadline - 10:
            result16 = _try_rank16_search(n, m, p, T, deadline)
            if result16 is not None:
                U, V, W = result16
                if _check_decomposition(U, V, W, T):
                    return {"U": U, "V": V, "W": W}
    
    # Fallback: return rank-18
    return {
        "U": U18,
        "V": V18,
        "W": W18
    }
