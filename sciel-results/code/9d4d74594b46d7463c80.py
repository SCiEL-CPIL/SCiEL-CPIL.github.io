"""Bootstrap SEARCH for rank-minimal exact tensor decomposition of 2x5 * 5x6."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _verify_rank(U, V, W, n=2, m=5, p=6):
    """Check if given factor matrices produce correct matmul tensor."""
    rank = len(U[0])
    
    # Check all on-diagonal entries equal 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check all off-diagonal entries equal 0
    for a in range(n*m):
        i = a // m
        j = a % m
        for b in range(m*p):
            j2 = b // p
            k = b % p
            if j != j2:
                continue
            for c in range(p*n):
                k2 = c // n
                i2 = c % n
                if k != k2 or i != i2:
                    val = 0.0
                    for cc in range(rank):
                        val += U[a][cc] * V[b][cc] * W[c][cc]
                    if abs(val) > 1e-9:
                        return False
    return True

def _schoolbook_rank():
    """Return the standard schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _als_solve(U, V, W, n=2, m=5, p=6, max_iter=100, tol=1e-10, deadline=None):
    """ALS for tensor decomposition. Returns (U, V, W, success)."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    U, V, W = U.astype(np.float64).copy(), V.astype(np.float64).copy(), W.astype(np.float64).copy()
    
    # Precompute target vectors
    tgt_W = np.zeros((pn, nm * mp))
    tgt_V = np.zeros((mp, pn * nm))
    tgt_U = np.zeros((nm, mp * pn))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a, b, c = i * m + j, j * p + k, k * n + i
                tgt_W[c, a * mp + b] = 1.0
                tgt_V[b, c * nm + a] = 1.0
                tgt_U[a, b * pn + c] = 1.0
    
    # Precompute off-diagonal mask for verification
    off_mask = np.zeros((nm, mp, pn), dtype=bool)
    for a in range(nm):
        ii, jj = a // m, a % m
        for b in range(mp):
            jj2, kk = b // p, b % p
            if jj != jj2:
                continue
            for c in range(pn):
                kk2, ii2 = c // n, c % n
                if kk != kk2 or ii != ii2:
                    off_mask[a, b, c] = True
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            return U, V, W, False
        
        # Solve for W
        KR = np.einsum('ar,br->ab r', U, V).reshape(nm * mp, rank)
        W_new = np.zeros((pn, rank))
        for c in range(pn):
            try:
                W_new[c] = np.linalg.lstsq(KR.T, tgt_W[c], rcond=None)[0]
            except:
                return U, V, W, False
        W = W_new
        
        # Solve for V
        KR = np.einsum('cr,ar->ac r', W, U).reshape(pn * nm, rank)
        V_new = np.zeros((mp, rank))
        for b in range(mp):
            try:
                V_new[b] = np.linalg.lstsq(KR.T, tgt_V[b], rcond=None)[0]
            except:
                return U, V, W, False
        V = V_new
        
        # Solve for U
        KR = np.einsum('br,cr->bc r', V, W).reshape(mp * pn, rank)
        U_new = np.zeros((nm, rank))
        for a in range(nm):
            try:
                U_new[a] = np.linalg.lstsq(KR.T, tgt_U[a], rcond=None)[0]
            except:
                return U, V, W, False
        U = U_new
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a, b, c = i * m + j, j * p + k, k * n + i
                    err = max(err, abs(np.dot(U[a] * V[b], W[c]) - 1.0))
        if err < tol:
            # Check off-diagonals
            off = 0.0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        if off_mask[a, b, c]:
                            off = max(off, abs(np.dot(U[a] * V[b], W[c])))
            if off < tol:
                return U, V, W, True
    return U, V, W, False

def _try_rank(r, n=2, m=5, p=6, n_trials=3, deadline=None):
    """Try rank-r decomposition via ALS."""
    nm, mp, pn = n * m, m * p, p * n
    for trial in range(n_trials):
        if deadline is not None and time.time() > deadline:
            break
        # Different initialization strategies
        if trial == 0:
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1
        elif trial == 1:
            # Sparse initialization
            U = np.zeros((nm, r))
            V = np.zeros((mp, r))
            W = np.zeros((pn, r))
            for c in range(r):
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
        else:
            # Mixed initialization
            U = np.random.randn(nm, r) * 0.1
            V = np.random.randn(mp, r) * 0.1
            W = np.random.randn(pn, r) * 0.1
            for c in range(min(r, 10)):
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
        
        U, V, W, ok = _als_solve(U, V, W, n, m, p, 80, 1e-9, deadline)
        if ok:
            res = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            if _verify_rank(res["U"], res["V"], res["W"], n, m, p):
                return res
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 130
    n, m, p = DIMENSIONS
    fb = _schoolbook_rank()
    
    # Try ranks below schoolbook (60)
    for r in [59, 58, 57]:
        if time.time() > deadline:
            break
        res = _try_rank(r, n, m, p, 3, deadline)
        if res is not None:
            return res
    
    return fb

# EVOLVE-BLOCK-END
