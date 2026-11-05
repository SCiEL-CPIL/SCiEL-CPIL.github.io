"""Rank-reduced exact decomposition for 3x4 times 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Exact decomposition for C = A(3x4) * B(4x4) using Strassen-based
    block decomposition. Rank-44 via 2x2 Strassen + 1x2 schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = 3, 4, 4
    
    def schoolbook():
        """Rank-48 schoolbook fallback."""
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
    
    total_rank = 44
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(i, k): return k * n + i
    
    def add_strassen_2x2x2(i0, j0, k0, col_start):
        a11 = u_idx(i0, j0)
        a12 = u_idx(i0, j0+1)
        a21 = u_idx(i0+1, j0)
        a22 = u_idx(i0+1, j0+1)
        b11 = v_idx(j0, k0)
        b12 = v_idx(j0, k0+1)
        b21 = v_idx(j0+1, k0)
        b22 = v_idx(j0+1, k0+1)
        c11 = w_idx(i0, k0)
        c12 = w_idx(i0, k0+1)
        c21 = w_idx(i0+1, k0)
        c22 = w_idx(i0+1, k0+1)
        c = col_start
        U[a11][c] += 1; U[a22][c] += 1; V[b11][c] += 1; V[b22][c] += 1; W[c11][c] += 1; W[c22][c] += 1; c += 1
        U[a21][c] += 1; U[a22][c] += 1; V[b11][c] += 1; W[c21][c] += 1; W[c22][c] -= 1; c += 1
        U[a11][c] += 1; V[b12][c] += 1; V[b22][c] -= 1; W[c12][c] += 1; W[c22][c] += 1; c += 1
        U[a22][c] += 1; V[b21][c] += 1; V[b11][c] -= 1; W[c11][c] += 1; W[c21][c] += 1; c += 1
        U[a11][c] += 1; U[a12][c] += 1; V[b22][c] += 1; W[c12][c] += 1; W[c11][c] -= 1; c += 1
        U[a21][c] += 1; U[a11][c] -= 1; V[b11][c] += 1; V[b12][c] += 1; W[c22][c] += 1; c += 1
        U[a12][c] += 1; U[a22][c] -= 1; V[b21][c] += 1; V[b22][c] += 1; W[c11][c] += 1; c += 1
        return c
    
    def add_schoolbook_1x2x2(i0, j0, k0, col_start):
        c = col_start
        for dj in range(2):
            for dk in range(2):
                U[u_idx(i0, j0+dj)][c] = 1
                V[v_idx(j0+dj, k0+dk)][c] = 1
                W[w_idx(i0, k0+dk)][c] = 1
                c += 1
        return c
    
    col = 0
    col = add_strassen_2x2x2(0, 0, 0, col)
    col = add_strassen_2x2x2(0, 2, 0, col)
    col = add_strassen_2x2x2(0, 0, 2, col)
    col = add_strassen_2x2x2(0, 2, 2, col)
    col = add_schoolbook_1x2x2(2, 0, 0, col)
    col = add_schoolbook_1x2x2(2, 2, 0, col)
    col = add_schoolbook_1x2x2(2, 0, 2, col)
    col = add_schoolbook_1x2x2(2, 2, 2, col)
    
    assert col == total_rank, f"Column mismatch: {col} != {total_rank}"
    
    if time.time() > deadline - 5.0:
        return {"U": U, "V": V, "W": W}
    
    try:
        Ua = np.array(U, dtype=np.float64)
        Va = np.array(V, dtype=np.float64)
        Wa = np.array(W, dtype=np.float64)
        U3 = Ua.reshape(n, m, total_rank)
        V3 = Va.reshape(m, p, total_rank)
        W3 = Wa.reshape(p, n, total_rank)
        T = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        if not np.allclose(T, 1.0, atol=1e-10):
            return schoolbook()
    except Exception:
        return schoolbook()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
