# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 150.0
_MARGIN = 10.0


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME) - _MARGIN


def _verify(U, V, W, tol=1e-7):
    n, m, p = 3, 3, 3
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _construct_strassen_block_decomposition():
    """
    Construct a decomposition by splitting 3x3 into:
    - Top-left 2x2 block: use Strassen (rank 7)
    - Bottom-right 1x1: direct (rank 1)  
    - Cross terms: handle with structured rank-16
    
    Total target rank: 24
    
    The 3x3 matrices are partitioned as:
    A = [[A11, A12],     B = [[B11, B12],     C = [[C11, C12],
         [A21, A22]]         [B21, B22]]         [C21, C22]]
    
    where A11,B11,C11 are 2x2, A12,B12,C12 are 2x1,
    A21,B21,C21 are 1x2, A22,B22,C22 are 1x1.
    
    Using Strassen for the 2x2 core and direct for cross terms.
    """
    n, m, p = 3, 3, 3
    
    # We'll build the decomposition term by term.
    # Each term r contributes U[.,r], V[.,r], W[.,r]
    
    terms = []  # list of (U_col, V_col, W_col)
    
    # Strassen formulas for 2x2 multiplication:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For the full 3x3, we need to handle:
    # C11 (2x2) = from A11,B11; A12,B21; A21,B12; A22,B22
    # C12 (2x1) = from A11,B12; A12,B22
    # C21 (1x2) = from A21,B11; A22,B21
    # C22 (1x1) = from A22,B22
    
    # Let's use a direct approach: enumerate all 27 products and
    # see if we can merge some using Strassen-like combinations.
    
    # Actually, let me try a cleaner construction:
    # Use the known result that 3x3 matrix mult has rank 23 (Landsberg 1988)
    # but let's first try to get a valid decomposition at rank 24 or 25.
    
    # Simplest valid approach: schoolbook is rank 27.
    # Let's try to reduce by using Strassen on the 2x2 sub-blocks.
    
    # I'll construct it explicitly with the following structure:
    # Terms 0-6: Strassen for top-left 2x2 (indices 0,1 for rows/cols)
    # Terms 7: A22*B22 (bottom-right corner)
    # Terms 8-23: cross terms (16 terms)
    
    # Let me define the index mapping:
    # Row index in A: i (0,1,2), Col index in A: j (0,1,2)
    # Row index in B: j (0,1,2), Col index in B: k (0,1,2)
    # Output: C[i,k]
    
    # U[i*m+j, r] = U[i*3+j, r]
    # V[j*p+k, r] = V[j*3+k, r]
    # W[k*n+i, r] = W[k*3+i, r]
    
    rank = 24
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Strassen for 2x2 block (indices 0,1)
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1  # A00*B00
    U[0, 0] += 1; V[3, 0] = 1; W[1, 0] = 1  # A00*B11 -> contributes to C01
    U[1, 0] += 1; V[0, 0] += 1; W[0, 0] += 1  # A10*B00 -> contributes to C10
    U[1, 0] += 1; V[3, 0] += 1; W[1, 0] += 1  # A11*B11 -> contributes to C11
    
    # This is getting complex. Let me use a different approach:
    # Just return schoolbook for now, but try to find a better one.
    
    # Actually, let me try a systematic construction.
    
    return None


def _schoolbook():
    n, m, p = 3, 3, 3
    U = np.zeros((n * m, n * m * p))
    V = np.zeros((m * p, n * m * p))
    W = np.zeros((p * n, n * m * p))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _try_fixed_UV_solve(rank):
    """Fix U,V as first `rank` schoolbook columns, solve for W, verify."""
    n, m, p = 3, 3, 3
    U_s, V_s, W_s = _schoolbook()
    if rank > U_s.shape[1]:
        return None
    U = U_s[:, :rank].copy()
    V = V_s[:, :rank].copy()
    
    num_vars = (p * n) * rank
    A = np.zeros((n * m * p, num_vars))
    b = np.ones(n * m * p)
    eq = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for r in range(rank):
                    var_idx = (k * n + i) * rank + r
                    A[eq, var_idx] = U[i * m + j, r] * V[j * p + k, r]
                eq += 1
    
    try:
        W_flat, residuals, rk, sv = np.linalg.lstsq(A, b, rcond=None)
        W = W_flat.reshape((p * n, rank))
    except Exception:
        return None
    
    if _verify(U, V, W):
        return U, V, W
    return None


def _general_search(rank, seed=0, max_attempts=10):
    """
    General search using scipy least_squares on the nonlinear system.
    """
    n, m, p = 3, 3, 3
    try:
        from scipy.optimize import least_squares
    except Exception:
        return None

    rng = np.random.default_rng(seed)
    nm, mp, pn = n * m, m * p, p * n

    def residuals(flat):
        U = flat[:nm * rank].reshape(nm, rank)
        V = flat[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W = flat[nm * rank + mp * rank:].reshape(pn, rank)
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    res[idx] = s - 1.0
                    idx += 1
        return res

    for attempt in range(max_attempts):
        if _time_left() < 8.0:
            break
        U0 = rng.normal(size=(nm, rank))
        V0 = rng.normal(size=(mp, rank))
        W0 = rng.normal(size=(pn, rank))
        scale = 1.0 / (rank ** 0.5)
        U0 *= scale
        V0 *= scale
        W0 *= scale
        flat0 = np.concatenate([U0.ravel(), V0.ravel(), W0.ravel()])
        try:
            sol = least_squares(residuals, flat0, method='lm', max_nfev=3000)
        except Exception:
            continue
        U = sol.x[:nm * rank].reshape(nm, rank)
        V = sol.x[nm * rank: nm * rank + mp * rank].reshape(mp, rank)
        W = sol.x[nm * rank + mp * rank:].reshape(pn, rank)
        if _verify(U, V, W, tol=1e-6):
            return U, V, W
    return None


def construct():
    n, m, p = 3, 3, 3

    # Start with schoolbook rank 27
    U_s, V_s, W_s = _schoolbook()
    best = (U_s, V_s, W_s)
    best_rank = 27

    # Try decreasing ranks
    for rank in range(26, 20, -1):
        if _time_left() < 20.0:
            break
        
        # First try fixed-U,V solve (fast)
        res = _try_fixed_UV_solve(rank)
        if res is not None:
            U, V, W = res
            if rank < best_rank:
                best = (U, V, W)
                best_rank = rank
            continue
        
        # Then try general search (slower)
        if _time_left() < 30.0:
            break
        res2 = _general_search(rank, seed=rank, max_attempts=5)
        if res2 is not None:
            U, V, W = res2
            if rank < best_rank:
                best = (U, V, W)
                best_rank = rank
            continue

    U, V, W = best
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
