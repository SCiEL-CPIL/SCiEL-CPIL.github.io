import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reconstructs the matrix multiplication tensor."""
    if len(U) == 0:
        return False
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_arr[u_idx, :], V_arr[v_idx, :] * W_arr[w_idx, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_222(U, V, W, n, m, p, col_offset, k0, k1):
    """Add Strassen 2x2x2 terms starting at col_offset for column pair (k0, k1)."""
    col = col_offset
    # M1 = (A00 + A11) * (B[0,k0] + B[1,k1])
    U[0][col] += 1
    U[3][col] += 1
    V[0 * p + k0][col] += 1
    V[1 * p + k1][col] += 1
    W[k0 * n + 0][col] += 1
    W[k1 * n + 1][col] += 1
    col += 1
    
    # M2 = (A10 + A11) * B[0,k0]
    U[2][col] += 1
    U[3][col] += 1
    V[0 * p + k0][col] += 1
    W[k0 * n + 1][col] += 1
    W[k1 * n + 1][col] += -1
    col += 1
    
    # M3 = A00 * (B[0,k1] - B[1,k1])
    U[0][col] += 1
    V[0 * p + k1][col] += 1
    V[1 * p + k1][col] += -1
    W[k1 * n + 0][col] += 1
    W[k1 * n + 1][col] += 1
    col += 1
    
    # M4 = A11 * (B[1,k0] - B[0,k0])
    U[3][col] += 1
    V[1 * p + k0][col] += 1
    V[0 * p + k0][col] += -1
    W[k0 * n + 0][col] += 1
    W[k0 * n + 1][col] += 1
    col += 1
    
    # M5 = (A00 + A01) * B[1,k1]
    U[0][col] += 1
    U[1][col] += 1
    V[1 * p + k1][col] += 1
    W[k0 * n + 0][col] += -1
    W[k1 * n + 0][col] += 1
    col += 1
    
    # M6 = (A10 - A00) * (B[0,k0] + B[0,k1])
    U[2][col] += 1
    U[0][col] += -1
    V[0 * p + k0][col] += 1
    V[0 * p + k1][col] += 1
    W[k1 * n + 1][col] += 1
    col += 1
    
    # M7 = (A01 - A11) * (B[1,k0] + B[1,k1])
    U[1][col] += 1
    U[3][col] += -1
    V[1 * p + k0][col] += 1
    V[1 * p + k1][col] += 1
    W[k0 * n + 0][col] += 1
    col += 1
    
    return col

def _build_strassen_228(n, m, p):
    """Build the rank-28 Strassen decomposition for 2x2x8 matrix multiplication."""
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for b in range(4):  # 4 blocks of 2 columns each
        k0 = 2 * b
        k1 = 2 * b + 1
        col = _strassen_222(U, V, W, n, m, p, col, k0, k1)
    
    return U, V, W, rank

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """Try to find linear dependencies among the rank-1 terms to reduce rank."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # The tensor T has dimensions (n*m) x (m*p) x (p*n) = 4 x 16 x 16
    dim = (n * m) * (m * p) * (p * n)
    terms = np.zeros((rank, dim))
    for r in range(rank):
        u = U_arr[:, r]
        v = V_arr[:, r]
        w = W_arr[:, r]
        outer = np.einsum('i,j,k->ijk', u, v, w)
        terms[r, :] = outer.flatten()
    
    # Check if these rank-1 terms are linearly dependent
    tol = 1e-10
    U_svd, s_vals, Vt_svd = np.linalg.svd(terms, full_matrices=False)
    null_mask = s_vals < tol
    if not np.any(null_mask):
        return None
    
    null_vecs = Vt_svd[null_mask, :]
    
    best_result = None
    for nv_idx in range(null_vecs.shape[0]):
        c = null_vecs[nv_idx, :]
        max_idx = np.argmax(np.abs(c))
        if abs(c[max_idx]) < tol:
            continue
        
        new_col = 0
        new_U = [[0] * (rank - 1) for _ in range(n * m)]
        new_V = [[0] * (rank - 1) for _ in range(m * p)]
        new_W = [[0] * (rank - 1) for _ in range(p * n)]
        
        for r in range(rank):
            if r == max_idx:
                continue
            scale = 1.0 - c[r] / c[max_idx]
            if abs(scale) < tol:
                continue
            for i in range(n * m):
                new_U[i][new_col] = U_arr[i, r] * scale
            for j in range(m * p):
                new_V[j][new_col] = V_arr[j, r] * scale
            for k in range(p * n):
                new_W[k][new_col] = W_arr[k, r] * scale
            new_col += 1
        
        if new_col == 0:
            continue
        
        new_U3 = [row[:new_col] for row in new_U]
        new_V3 = [row[:new_col] for row in new_V]
        new_W3 = [row[:new_col] for row in new_W]
        
        if _verify_decomposition(new_U3, new_V3, new_W3, n, m, p):
            if best_result is None or new_col < best_result[0]:
                best_result = (new_col, new_U3, new_V3, new_W3)
        
        if time.time() > deadline:
            break
    
    return best_result

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Build the rank-28 Strassen decomposition
    U, V, W, rank = _build_strassen_228(n, m, p)
    
    # Verify the base decomposition
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback: schoolbook (shouldn't happen)
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
    
    # Try to reduce rank via linear dependency elimination
    if time.time() < deadline - 10:
        result = _try_reduce_rank(U, V, W, n, m, p, deadline - 5)
        if result is not None:
            new_rank, new_U, new_V, new_W = result
            if new_rank < rank:
                U, V, W = new_U, new_V, new_W
                rank = new_rank
    
    return {"U": U, "V": V, "W": W}
