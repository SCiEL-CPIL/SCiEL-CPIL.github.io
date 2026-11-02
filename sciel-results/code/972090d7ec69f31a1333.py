# EVOLVE-BLOCK-START
import time
import numpy as np


def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    for M in (U, V, W):
        if np.max(np.abs(M * 2 - np.round(M * 2))) > tol:
            return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.einsum('c,c,c->', U[i * m + j], V[j * p + k], W[k * n + i])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _build_schoolbook(n, m, p):
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


def _to_int_lists(U, V, W):
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    def conv(M):
        M = np.round(M * 2) / 2.0
        out = []
        for row in M:
            out.append([float(x) for x in row])
        return out
    return {"U": conv(U), "V": conv(V), "W": conv(W)}


def _get_strassen_2x2x2():
    """Verified 2x2x2 Strassen rank-7 decomposition."""
    # These are the 7 rank-1 tensors for 2x2x2 matmul.
    # u: A entries [a,b,c,d], v: B entries [e,f,g,h], w: C entries [C00,C01,C10,C11]
    # where C = A*B, C[i,k] = sum_j A[i,j]*B[j,k]
    # w index: k*2+i (per the convention W[k*n+i])
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]),
        ([0, 0, 0, 1], [0, 0, 1, -1], [-1, 1, 0, 0]),
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
        ([0, 0, 1, -1], [1, 0, 1, 0], [0, 0, 0, 1]),
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, -1]),
    ]
    # Verify
    T = np.zeros((2, 2, 2))
    for u, v, w in terms:
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T[i, j, k] += u[i*2+j] * v[j*2+k] * w[k*2+i]
    if np.max(np.abs(T - 1.0)) > 1e-9:
        return None
    return terms


def _get_strassen_2x2x3():
    """Rank-9 decomposition for 2x2 x 2x3 matmul."""
    s22 = _get_strassen_2x2x2()
    if s22 is None:
        return None
    terms = []
    for u, v, w in s22:
        # Extend v from 2x2 (4 entries) to 2x3 (6 entries): add 0 for 3rd col
        v6 = v + [0]
        # Extend w from 2x2 (4 entries: k*2+i for k=0,1) to 2x3 (6 entries: k*2+i for k=0,1,2)
        # The 2x2 w covers k=0,1. Add 0 for k=2.
        w6 = w + [0, 0]
        terms.append((u, v6, w6))
    # Add schoolbook terms for 3rd column of C
    # C[0,2] = a*B[0,2] + b*B[1,2], C[1,2] = c*B[0,2] + d*B[1,2]
    terms.append(([1,0,0,0], [0,0,1,0,0,0], [0,0,0,0,1,0]))
    terms.append(([0,1,0,0], [0,0,0,0,0,1], [0,0,0,0,1,0]))
    terms.append(([0,0,1,0], [0,0,1,0,0,0], [0,0,0,0,0,1]))
    terms.append(([0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1]))
    # Verify
    T = np.zeros((2, 2, 3))
    for u, v, w in terms:
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    T[i, j, k] += u[i*2+j] * v[j*3+k] * w[k*2+i]
    if np.max(np.abs(T - 1.0)) > 1e-9:
        return None
    return terms


def _build_4x4x4x5_rank64():
    """Rank-64 construction: 4x4x4x2 (rank 28) + 4x4x4x3 (rank 36)."""
    s22 = _get_strassen_2x2x2()
    s23 = _get_strassen_2x2x3()
    if s22 is None or s23 is None:
        return None
    
    n, m, p = 4, 4, 5
    r28 = 4 * 7   # 4 products of 2x2x2
    r36 = 4 * 9   # 4 products of 2x2x3
    r = r28 + r36
    
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    col = 0
    # 4x4 x 4x2 part (B cols 0-1, C cols 0-1)
    for i_blk in range(2):
        for k_blk in range(2):
            for (u, v, w) in s22:
                for a_idx in range(4):
                    ar, ac = divmod(a_idx, 2)
                    U[(2*i_blk+ar)*m + (2*k_blk+ac), col] = u[a_idx]
                for v_idx in range(4):
                    br, bc = divmod(v_idx, 2)
                    V[(2*k_blk+br)*p + bc, col] = v[v_idx]
                for w_idx in range(4):
                    cr, cc = divmod(w_idx, 2)
                    W[cr*n + (2*i_blk+cc), col] = w[w_idx]
                col += 1
    
    # 4x4 x 4x3 part (B cols 2-4, C cols 2-4)
    for i_blk in range(2):
        for k_blk in range(2):
            for (u, v, w) in s23:
                for a_idx in range(4):
                    ar, ac = divmod(a_idx, 2)
                    U[(2*i_blk+ar)*m + (2*k_blk+ac), col] = u[a_idx]
                for v_idx in range(6):
                    br, bc = divmod(v_idx, 3)
                    V[(2*k_blk+br)*p + (bc+2), col] = v[v_idx]
                for w_idx in range(6):
                    cr, cc = divmod(w_idx, 2)
                    W[(cr+2)*n + (2*i_blk+cc), col] = w[w_idx]
                col += 1
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140
    
    best = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    
    candidates = []
    if time.time() < deadline - 5:
        try:
            cand = _build_4x4x4x5_rank64()
            if cand is not None:
                candidates.append(cand)
        except Exception:
            pass
    
    for Uf, Vf, Wf in candidates:
        if time.time() > deadline - 2:
            break
        try:
            if _verify(Uf, Vf, Wf, n, m, p):
                r = Uf.shape[1]
                if r < best_rank:
                    best = _to_int_lists(Uf, Vf, Wf)
                    best_rank = r
        except Exception:
            pass
    
    return best
# EVOLVE-BLOCK-END
