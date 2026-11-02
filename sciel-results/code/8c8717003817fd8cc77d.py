import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    Texp = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                Texp[a, b, c] = 1.0
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        T += np.einsum('a,b,c->abc', U[:,r], V[:,r], W[:,r])
    err = np.max(np.abs(T - Texp))
    return err < tol

def _schoolbook(n, m, p):
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

def _find_223_decomp(rank=10, max_iters=200, seed=42):
    """Try to find a rank-r decomposition for 2x2x3 matrix multiplication tensor."""
    rng = np.random.RandomState(seed)
    
    # Tensor: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # A: 4 entries (i*2+j), B: 6 entries (j*3+k), C: 6 entries (k*2+i)
    Texp = np.zeros((4, 6, 6))
    for i in range(2):
        for j in range(2):
            for k in range(3):
                a = i*2 + j
                b = j*3 + k
                c = k*2 + i
                Texp[a, b, c] = 1.0
    
    best_err = np.inf
    best_U, best_V, best_W = None, None, None
    
    for attempt in range(max_iters):
        U = rng.randn(4, rank)
        V = rng.randn(6, rank)
        W = rng.randn(6, rank)
        
        # Normalize each column
        for r in range(rank):
            nu = np.linalg.norm(U[:, r])
            nv = np.linalg.norm(V[:, r])
            nw = np.linalg.norm(W[:, r])
            if nu > 1e-15: U[:, r] /= nu
            if nv > 1e-15: V[:, r] /= nv
            if nw > 1e-15: W[:, r] /= nw
        
        T = np.einsum('ar,br,cr->abc', U, V, W)
        err = np.max(np.abs(T - Texp))
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if err < 1e-9:
            break
    
    if best_err < 1e-9:
        return best_U, best_V, best_W
    return None

def _strassen_222_block(U, V, W, col, i0, j0, k0, m, p, n):
    a_idx = i0 * m + j0
    b_idx = i0 * m + (j0 + 1)
    c_idx = (i0 + 1) * m + j0
    d_idx = (i0 + 1) * m + (j0 + 1)
    e_idx = j0 * p + k0
    f_idx = j0 * p + (k0 + 1)
    g_idx = (j0 + 1) * p + k0
    h_idx = (j0 + 1) * p + (k0 + 1)
    c00_idx = k0 * n + i0
    c01_idx = (k0 + 1) * n + i0
    c10_idx = k0 * n + (i0 + 1)
    c11_idx = (k0 + 1) * n + (i0 + 1)
    
    U[a_idx, col] += 1; U[d_idx, col] += 1
    V[e_idx, col] += 1; V[h_idx, col] += 1
    W[c00_idx, col] += 1; W[c11_idx, col] += 1
    col += 1
    U[c_idx, col] += 1; U[d_idx, col] += 1
    V[e_idx, col] += 1
    W[c10_idx, col] += 1; W[c11_idx, col] -= 1
    col += 1
    U[a_idx, col] += 1
    V[f_idx, col] += 1; V[h_idx, col] -= 1
    W[c01_idx, col] += 1; W[c11_idx, col] += 1
    col += 1
    U[d_idx, col] += 1
    V[g_idx, col] += 1; V[e_idx, col] -= 1
    W[c00_idx, col] += 1; W[c10_idx, col] += 1
    col += 1
    U[a_idx, col] += 1; U[b_idx, col] += 1
    V[h_idx, col] += 1
    W[c00_idx, col] -= 1; W[c01_idx, col] += 1
    col += 1
    U[c_idx, col] += 1; U[a_idx, col] -= 1
    V[e_idx, col] += 1; V[f_idx, col] += 1
    W[c11_idx, col] += 1
    col += 1
    U[b_idx, col] += 1; U[d_idx, col] -= 1
    V[g_idx, col] += 1; V[h_idx, col] += 1
    W[c00_idx, col] += 1
    col += 1
    return col

def _schoolbook_221_block(U, V, W, col, i0, j0, k0, m, p, n):
    a_idx = i0 * m + j0
    b_idx = i0 * m + (j0 + 1)
    c_idx = (i0 + 1) * m + j0
    d_idx = (i0 + 1) * m + (j0 + 1)
    e_idx = j0 * p + k0
    g_idx = (j0 + 1) * p + k0
    c00_idx = k0 * n + i0
    c10_idx = k0 * n + (i0 + 1)
    
    U[a_idx, col] += 1; V[e_idx, col] += 1; W[c00_idx, col] += 1
    col += 1
    U[b_idx, col] += 1; V[g_idx, col] += 1; W[c00_idx, col] += 1
    col += 1
    U[c_idx, col] += 1; V[e_idx, col] += 1; W[c10_idx, col] += 1
    col += 1
    U[d_idx, col] += 1; V[g_idx, col] += 1; W[c10_idx, col] += 1
    col += 1
    return col

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110
    
    # Try rank-10 for 2x2x3 to get total rank 48
    decomp_223 = None
    if time.time() < deadline - 80:
        for seed in range(30):
            if time.time() > deadline - 80:
                break
            result = _find_223_decomp(rank=10, max_iters=100, seed=seed)
            if result is not None:
                decomp_223 = result
                break
    
    if decomp_223 is not None:
        U223, V223, W223 = decomp_223
        rank_223 = U223.shape[1]
        total_rank = 2 * (2 * rank_223 + 4)
        
        U = np.zeros((n * m, total_rank), dtype=np.float64)
        V = np.zeros((m * p, total_rank), dtype=np.float64)
        W = np.zeros((p * n, total_rank), dtype=np.float64)
        
        col = 0
        for j in range(2):
            for k_start in [0, 3]:
                for r in range(rank_223):
                    for i in range(2):
                        for jj in range(2):
                            U[i * m + (j + jj), col] += U223[i*2+jj, r]
                        for jj in range(2):
                            for kk in range(3):
                                V[(j + jj) * p + (k_start + kk), col] += V223[jj*3+kk, r]
                        for kk in range(3):
                            for i in range(2):
                                W[(k_start + kk) * n + i, col] += W223[kk*2+i, r]
                    col += 1
            # Schoolbook for k=6
            for i in range(2):
                for jj in range(2):
                    U[i * m + (j + jj), col] += 1
                    V[(j + jj) * p + 6, col] += 1
                    W[6 * n + i, col] += 1
                    col += 1
        
        if col != total_rank:
            pass  # fall through to rank-50
        elif _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Rank-50 fallback
    rank = 50
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    col = 0
    
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            col = _strassen_222_block(U, V, W, col, 0, j_start, k_start, m, p, n)
        col = _schoolbook_221_block(U, V, W, col, 0, j_start, 6, m, p, n)
    
    if col != rank:
        return _schoolbook(n, m, p)
    
    result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    if _verify(U, V, W, n, m, p):
        return result
    
    return _schoolbook(n, m, p)
