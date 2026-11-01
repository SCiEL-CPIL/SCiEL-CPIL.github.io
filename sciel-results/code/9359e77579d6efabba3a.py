"""Combined: rank-18 base with enhanced rank-17 search using structured init and polish."""
import numpy as np
import time

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
    """Check if sum_r U_u,r * V_v,r * W_w,r == T_u,v,w for all u,v,w."""
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

def _compute_error(U, V, W, T, n, m, p):
    """Compute RMS error of the decomposition."""
    err = 0.0
    for r in range(U.shape[1]):
        err += np.sum((np.einsum('u,v,w->uvw', U[:,r], V[:,r], W[:,r]) - T)**2)
    return np.sqrt(err / (n*m * m*p * p*n))

def _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, R, max_iters=200, tol=1e-13, deadline=None):
    """Run ALS from a given initialization, return (U, V, W) as lists if exact."""
    U = np.array(U_init, dtype=np.float64)
    V = np.array(V_init, dtype=np.float64)
    W = np.array(W_init, dtype=np.float64)
    T_a = np.array(T, dtype=np.float64)
    
    prev_err = np.inf
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        U = _als_solve_U(T_a, V, W, n, m, p, R)
        V = _als_solve_V(T_a, U, W, n, m, p, R)
        W = _als_solve_W(T_a, U, V, n, m, p, R)
        
        err = _compute_error(U, V, W, T_a, n, m, p)
        
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    
    if _check_decomposition(U, V, W, T_a):
        # Try to snap to half-integers
        for denom in [2, 4, 8, 16, 32, 64]:
            U_s = np.round(U * denom) / denom
            V_s = np.round(V * denom) / denom
            W_s = np.round(W * denom) / denom
            if _check_decomposition(U_s, V_s, W_s, T_a):
                return U_s.tolist(), V_s.tolist(), W_s.tolist()
        return U.tolist(), V.tolist(), W.tolist()
    return None

def _structured_rank17_init(n, m, p, seed):
    """Generate structured rank-17 initializations that respect problem symmetry."""
    np.random.seed(seed)
    
    # Base structure: use Strassen-like patterns for 4 pairs of columns
    # with one shared term
    U = np.zeros((4, 17))
    V = np.zeros((10, 17))
    W = np.zeros((10, 17))
    
    # Pattern 1: Full Strassen for cols 0-3 (14 terms), then 3 for col 4
    # But we need to merge one term
    
    # Try various structured patterns
    pattern = seed % 5
    
    if pattern == 0:
        # Standard Strassen + 3 schoolbook, with one Strassen term merged
        # Cols 0,1: 7 Strassen terms
        for r in range(7):
            # Standard Strassen entries with small random perturbation
            pass  # Will be filled below
        
        # Use the known rank-18 structure but merge terms 0 and 6
        U18, V18, W18 = _build_rank18(n, m, p)
        U18_a = np.array(U18, dtype=np.float64)
        V18_a = np.array(V18, dtype=np.float64)
        W18_a = np.array(W18, dtype=np.float64)
        
        # Merge columns 0 and 6 by averaging
        idx = [i for i in range(18) if i not in [0, 6]]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        # Add a new merged term
        U_init[:, 0] = (U18_a[:, 0] + U18_a[:, 6]) / 2
        V_init[:, 0] = (V18_a[:, 0] + V18_a[:, 6]) / 2
        W_init[:, 0] = (W18_a[:, 0] + W18_a[:, 6]) / 2
        
        # Perturb
        scale = 0.02 + 0.01 * (seed % 3)
        U_init += np.random.randn(4, 17) * scale
        V_init += np.random.randn(10, 17) * scale
        W_init += np.random.randn(10, 17) * scale
        
    elif pattern == 1:
        # Merge schoolbook terms 14 and 16
        U18, V18, W18 = _build_rank18(n, m, p)
        U18_a = np.array(U18, dtype=np.float64)
        V18_a = np.array(V18, dtype=np.float64)
        W18_a = np.array(W18, dtype=np.float64)
        
        idx = [i for i in range(18) if i not in [14, 16]]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        U_init[:, 0] = (U18_a[:, 14] + U18_a[:, 16]) / 2
        V_init[:, 0] = (V18_a[:, 14] + V18_a[:, 16]) / 2
        W_init[:, 0] = (W18_a[:, 14] + W18_a[:, 16]) / 2
        
        scale = 0.02 + 0.01 * (seed % 3)
        U_init += np.random.randn(4, 17) * scale
        V_init += np.random.randn(10, 17) * scale
        W_init += np.random.randn(10, 17) * scale
        
    elif pattern == 2:
        # Random but with structured sparsity
        U_init = np.random.randn(4, 17) * 0.3
        V_init = np.random.randn(10, 17) * 0.3
        W_init = np.random.randn(10, 17) * 0.3
        
        # Make some entries zero to encourage structure
        for _ in range(10):
            U_init[np.random.randint(4), np.random.randint(17)] = 0
        for _ in range(20):
            V_init[np.random.randint(10), np.random.randint(17)] = 0
        for _ in range(20):
            W_init[np.random.randint(10), np.random.randint(17)] = 0
            
    elif pattern == 3:
        # Use symmetry: U[0] and U[3] should be related
        U_init = np.random.randn(4, 17) * 0.4
        U_init[3] = U_init[0] + np.random.randn(17) * 0.1  # Related
        U_init[1] = -U_init[2] + np.random.randn(17) * 0.1  # Antisymmetric hint
        
        V_init = np.random.randn(10, 17) * 0.4
        W_init = np.random.randn(10, 17) * 0.4
        
    else:
        # Pure random with varied scale
        scale = 0.2 + 0.3 * (seed % 4) / 4
        U_init = np.random.randn(4, 17) * scale
        V_init = np.random.randn(10, 17) * scale
        W_init = np.random.randn(10, 17) * scale
    
    return U_init, V_init, W_init

def _try_rank17_search(n, m, p, T, deadline):
    """Try to find a rank-17 decomposition with enhanced strategies."""
    
    # Strategy 1: Structured initializations
    for seed in range(60):
        if time.time() > deadline - 5:
            break
        U_init, V_init, W_init = _structured_rank17_init(n, m, p, seed)
        result = _try_rank_als_from_init(U_init, V_init, W_init, T, n, m, p, 17, 
                                          max_iters=150, deadline=deadline - 2)
        if result is not None:
            return result
    
    # Strategy 2: Remove each term from rank-18 and perturb
    U18, V18, W18 = _build_rank18(n, m, p)
    U18_a = np.array(U18, dtype=np.float64)
    V18_a = np.array(V18, dtype=np.float64)
    W18_a = np.array(W18, dtype=np.float64)
    
    for term_to_remove in range(18):
        if time.time() > deadline - 5:
            break
        
        idx = [i for i in range(18) if i != term_to_remove]
        U_init = U18_a[:, idx].copy()
        V_init = V18_a[:, idx].copy()
        W_init = W18_a[:, idx].copy()
        
        for seed in range(3):
            if time.time() > deadline - 2:
                break
            np.random.seed(term_to_remove * 100 + seed)
            scale = 0.01 + 0.005 * (seed % 3)
            U_pert = U_init + np.random.randn(4, 17) * scale
            V_pert = V_init + np.random.randn(10, 17) * scale
            W_pert = W_init + np.random.randn(10, 17) * scale
            
            result = _try_rank_als_from_init(U_pert, V_pert, W_pert, T, n, m, p, 17, 
                                              max_iters=120, deadline=deadline - 1)
            if result is not None:
                return result
    
    return None

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
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
    
    # Fallback: return rank-18
    return {
        "U": U18,
        "V": V18,
        "W": W18
    }
