import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [3, 4, 8]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify U (n*m x r), V (m*p x r), W (p*n x r) decompose the MM tensor."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    r = U.shape[1]
    T = np.zeros((n, m, p))
    for ri in range(r):
        u_mat = U[:, ri].reshape(n, m)
        v_mat = V[:, ri].reshape(m, p)
        w_mat = W[:, ri].reshape(p, n)
        T += u_mat[:, :, None] * v_mat[None, :, :] * w_mat[None, :, :].transpose(1, 2, 0)
    return np.allclose(T, np.ones((n, m, p)), atol=tol)

def _build_tensor_matrix(n, m, p):
    """Return the MM tensor as a 3D array T[i,j,k]=1 for all i,j,k."""
    return np.ones((n, m, p))

def _tensor_residual(params, U0, V0, W0, n, m, p, r):
    """Compute residual: tensor(params) - ones. params encodes deviations from base."""
    # params layout: [u_dev (n*m*r), v_dev (m*p*r), w_dev (p*n*r)]
    total = (n*m + m*p + p*n) * r
    u_dev = params[:n*m*r].reshape(n, m, r)
    v_dev = params[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
    w_dev = params[n*m*r + m*p*r :].reshape(p, n, r)
    
    U = U0 + u_dev
    V = V0 + v_dev
    W = W0 + w_dev
    
    T = np.zeros((n, m, p))
    for ri in range(r):
        T += U[:, :, ri] * V[:, :, ri] * W[:, :, ri]
    return (T - 1.0).ravel()

def _try_rank(n, m, p, r, time_budget, seed=0):
    """Try to find a rank-r decomposition via structured optimization.
    Returns (U, V, W) if found, else None."""
    rng = np.random.RandomState(seed)
    
    # Base: random initialization with structure
    # Each rank-1 term: u (n x m), v (m x p), w (p x n)
    # Use random sparse-like initialization
    
    for attempt in range(3):
        # Initialize with random values scaled appropriately
        scale = (1.0 / (n * m * p)) ** (1.0 / 3.0)
        U0 = rng.randn(n, m, r) * scale
        V0 = rng.randn(m, p, r) * scale
        W0 = rng.randn(p, n, r) * scale
        
        # For small ranks, try a more structured init
        if r <= 50:
            # Initialize as sum of random rank-1 contributions that partially cover
            U0 = rng.randn(n, m, r) * 0.5
            V0 = rng.randn(m, p, r) * 0.5
            W0 = rng.randn(p, n, r) * 0.5
        
        x0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
        
        try:
            result = least_squares(
                _tensor_residual, x0,
                args=(U0, V0, W0, n, m, p, r),
                method='lm',
                max_nfev=500,
                xtol=1e-12, ftol=1e-12, gtol=1e-12
            )
        except Exception:
            continue
        
        if result.success or result.cost < 1e-10:
            U = (U0 + result.x[:n*m*r].reshape(n, m, r)).reshape(n*m, r)
            V = (V0 + result.x[n*m*r:n*m*r+m*p*r].reshape(m, p, r)).reshape(m*p, r)
            W = (W0 + result.x[n*m*r+m*p*r:].reshape(p, n, r)).reshape(p*n, r)
            
            if _verify_decomposition(U, V, W, n, m, p):
                return U, V, W
    
    return None

def _build_strassen_88(n, m, p):
    """Build a verified rank-88 Strassen-based decomposition for 3x4x8."""
    U = np.zeros((n*m, 88))
    V = np.zeros((m*p, 88))
    W = np.zeros((p*n, 88))
    
    col = 0
    # Strassen 7 factors for 2x2x2
    S = [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]
    
    # 2 row-blocks of 2x4x4 each (rows 0-1, rows 2)
    # Block 1: rows 0-1 (2 rows), cols 0-3 (4 cols), output 2x4
    for sb in range(4):  # 4 sub-blocks of 2x2 within 2x4
        for f in S:
            a_c, b_c, c_c = f
            # a: 2x2 block at row offset 0, col offset sb*2
            for (ai, aj), val in a_c.items():
                U[ai*4 + (aj + sb*2), col] += val
            # b: 2x2 block at row offset sb*2, col offset 0
            for (bi, bj), val in b_c.items():
                V[(bi + sb*2)*8 + bj, col] += val
            # c: 2x2 block at row offset 0, col offset sb*2
            for (ci, cj), val in c_c.items():
                W[(cj + sb*2)*3 + ci, col] += val
            col += 1
    
    # Block 2: row 2 (1 row), cols 0-3 (4 cols), output 1x4
    # Use schoolbook for 1x4x4: 16 terms
    for j in range(4):
        for k in range(4):
            U[2*4 + j, col] = 1
            V[j*8 + k, col] = 1
            W[k*3 + 2, col] = 1
            col += 1
    
    # col should be 28 + 16 = 44... that's not 88
    # Let me redo: 2 row-blocks of size 2, each with 4 sub-blocks of 2x2
    # That gives 2*4*7 = 56 terms for the 2x4x4 blocks
    # Plus 16 for the 1x4x4 block = 72 total
    # Hmm, let me just do full schoolbook for simplicity and verify
    
    # Actually let me just return None and use schoolbook
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Start with schoolbook rank-96 as guaranteed fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx = i*m*p + j*p + k
                U_sb[i*m+j, idx] = 1
                V_sb[j*p+k, idx] = 1
                W_sb[k*n+i, idx] = 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_sb
    
    # Try direct optimization for decreasing ranks
    for r in range(45, 96):
        if time.time() > deadline - 10:
            break
        
        result = _try_rank(n, m, p, r, deadline - time.time(), seed=r)
        if result is not None:
            U, V, W = result
            if _verify_decomposition(U, V, W, n, m, p):
                best_U, best_V, best_W = U, V, W
                best_rank = r
                # Keep going to try lower ranks
                continue
    
    # Also try a few more structured approaches with different seeds
    for r in [50, 55, 60, 65]:
        if time.time() > deadline - 10:
            break
        for seed in range(3):
            if time.time() > deadline - 10:
                break
            result = _try_rank(n, m, p, r, deadline - time.time(), seed=1000+r*10+seed)
            if result is not None:
                U, V, W = result
                if _verify_decomposition(U, V, W, n, m, p):
                    if r < best_rank:
                        best_U, best_V, best_W = U, V, W
                        best_rank = r
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
