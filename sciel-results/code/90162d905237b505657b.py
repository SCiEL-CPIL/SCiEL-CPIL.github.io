"""
Matrix multiplication 2x4x7 via tensor decomposition.
Optimized search for rank < 50 using structural analysis of the tensor.
"""

import time
import numpy as np

_EVAL_DEADLINE = 145.0
_START = time.time()

def _time_left():
    return _EVAL_DEADLINE - time.time()

def _build_tensor(n, m, p):
    """Build multiplication tensor T[r, i, j] where r=k*n+i, j=row_of_B*p+col_of_A"""
    r_dim = p * n
    T = np.zeros((r_dim, n, m * p))
    for k in range(p):
        for i in range(n):
            for j in range(m):
                r = k * n + i
                jj = j * p + k
                T[r, i, jj] = 1.0
    return T

def _construct_rank50(n=2, m=4, p=7):
    """Known rank-50 construction using Strassen blocks + direct terms."""
    rank = 50
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = [0]

    def add_strassen(i0, j0, k0):
        c = col[0]
        # M1: (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        U[i0*m + j0, c] += 1; U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1; V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1; W[(k0+1)*n + (i0+1), c] += 1
        c += 1
        # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0+1)*m + j0, c] += 1; U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        W[k0*n + (i0+1), c] += 1; W[(k0+1)*n + (i0+1), c] -= 1
        c += 1
        # M3: A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        U[i0*m + j0, c] += 1
        V[j0*p + (k0+1), c] += 1; V[(j0+1)*p + (k0+1), c] -= 1
        W[(k0+1)*n + i0, c] += 1; W[(k0+1)*n + (i0+1), c] += 1
        c += 1
        # M4: A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        U[(i0+1)*m + (j0+1), c] += 1
        V[(j0+1)*p + k0, c] += 1; V[j0*p + k0, c] -= 1
        W[k0*n + i0, c] += 1; W[k0*n + (i0+1), c] += 1
        c += 1
        # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0*m + j0, c] += 1; U[i0*m + (j0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] -= 1; W[(k0+1)*n + i0, c] += 1
        c += 1
        # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        U[(i0+1)*m + j0, c] += 1; U[i0*m + j0, c] -= 1
        V[j0*p + k0, c] += 1; V[j0*p + (k0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += 1
        c += 1
        # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        U[i0*m + (j0+1), c] += 1; U[(i0+1)*m + (j0+1), c] -= 1
        V[(j0+1)*p + k0, c] += 1; V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1
        col[0] = c

    # 2x4x4 block: four 2x2x2 Strassen blocks (k=0,1,2,3)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    # 2x4x2 block: two 2x2x2 Strassen blocks (k=4,5)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    # 2x4x1 block: 8 direct rank-1 terms (k=6)
    for i in range(n):
        for j in range(m):
            U[i*m + j, col[0]] = 1.0
            V[j*p + 6, col[0]] = 1.0
            W[6*n + i, col[0]] = 1.0
            col[0] += 1

    return {"U": U, "V": V, "W": W}

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check that factor matrices exactly reproduce the tensor."""
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    T = _build_tensor(n, m, p)
    T_mat = T.reshape(r_dim, i_dim * j_dim)
    rank = W.shape[1]
    cand = np.zeros_like(T_mat)
    for t in range(rank):
        wt = W[:, t]
        ut = U[:, t]
        vt = V[:, t]
        outer = np.outer(ut, vt).flatten()
        cand += np.outer(wt, outer)
    return np.max(np.abs(cand - T_mat)) < tol

def _try_reduce_rank50(n=2, m=4, p=7):
    """Try to find rank-49 by identifying linear dependence among 50 terms."""
    base = _construct_rank50(n, m, p)
    U_base, V_base, W_base = base["U"], base["V"], base["W"]
    r_dim = p * n
    i_dim = n
    j_dim = m * p
    T = _build_tensor(n, m, p)
    T_mat = T.reshape(r_dim, i_dim * j_dim)

    # Build term matrix: each column is a rank-1 term flattened
    term_cols = []
    for t in range(50):
        term = np.outer(W_base[:, t], np.outer(U_base[:, t], V_base[:, t]).flatten())
        term_cols.append(term)
    term_matrix = np.column_stack(term_cols)

    # Check rank of term matrix
    try:
        rank_tm = np.linalg.matrix_rank(term_matrix, tol=1e-10)
        if rank_tm < 50:
            # Find which term to remove
            for t in range(50):
                if _time_left() < 5:
                    break
                mask = [k for k in range(50) if k != t]
                others = term_matrix[:, mask]
                if np.linalg.matrix_rank(others, tol=1e-10) == rank_tm:
                    U_new = np.delete(U_base, t, axis=1)
                    V_new = np.delete(V_base, t, axis=1)
                    W_new = np.delete(W_base, t, axis=1)
                    cand = np.zeros_like(T_mat)
                    for tt in range(49):
                        cand += np.outer(W_new[:, tt], np.outer(U_new[:, tt], V_new[:, tt]).flatten())
                    if np.max(np.abs(cand - T_mat)) < 1e-10:
                        return {"U": U_new, "V": V_new, "W": W_new}
    except:
        pass
    return None

def _direct_construction(n=2, m=4, p=7):
    """Direct block construction with optimized 2x4x2 and 2x4x1 blocks."""
    # Use Strassen for 2x4x4 (rank 28), then optimize remaining 2x4x3 (k=4,5,6)
    rank_2x4x4 = 28  # four 2x2x2 Strassen blocks
    # For 2x4x3: use mixed approach
    # k=4,5: can use 2x2x2 blocks but with sharing
    # k=6: direct terms
    
    # Try rank 49 by combining blocks optimally
    U = np.zeros((n * m, 49))
    V = np.zeros((m * p, 49))
    W = np.zeros((p * n, 49))

    col = [0]

    def add_strassen(i0, j0, k0):
        c = col[0]
        U[i0*m + j0, c] += 1; U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1; V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1; W[(k0+1)*n + (i0+1), c] += 1
        c += 1
        U[(i0+1)*m + j0, c] += 1; U[(i0+1)*m + (j0+1), c] += 1
        V[j0*p + k0, c] += 1
        W[k0*n + (i0+1), c] += 1; W[(k0+1)*n + (i0+1), c] -= 1
        c += 1
        U[i0*m + j0, c] += 1
        V[j0*p + (k0+1), c] += 1; V[(j0+1)*p + (k0+1), c] -= 1
        W[(k0+1)*n + i0, c] += 1; W[(k0+1)*n + (i0+1), c] += 1
        c += 1
        U[(i0+1)*m + (j0+1), c] += 1
        V[(j0+1)*p + k0, c] += 1; V[j0*p + k0, c] -= 1
        W[k0*n + i0, c] += 1; W[k0*n + (i0+1), c] += 1
        c += 1
        U[i0*m + j0, c] += 1; U[i0*m + (j0+1), c] += 1
        V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] -= 1; W[(k0+1)*n + i0, c] += 1
        c += 1
        U[(i0+1)*m + j0, c] += 1; U[i0*m + j0, c] -= 1
        V[j0*p + k0, c] += 1; V[j0*p + (k0+1), c] += 1
        W[(k0+1)*n + (i0+1), c] += 1
        c += 1
        U[i0*m + (j0+1), c] += 1; U[(i0+1)*m + (j0+1), c] -= 1
        V[(j0+1)*p + k0, c] += 1; V[(j0+1)*p + (k0+1), c] += 1
        W[k0*n + i0, c] += 1
        col[0] = c

    # 2x4x4: four Strassen blocks
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    # Now col=28, need 21 more for k=4,5,6
    
    # 2x4x2 (k=4,5): use 2 Strassen blocks (rank 14) -> col=42
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    # Now col=42, need 7 more for k=6
    
    # 2x4x1 (k=6): try rank 7 instead of 8
    # Direct terms: for each (i,j), need U[i*4+j] * V[j*7+6] * W[12+i] = 1
    # Can we do this with 7 terms?
    # Try: use 4 terms for i=0, 3 terms for i=1 with sharing
    for j in range(m):
        U[0*m + j, col[0]] = 1.0
        V[j*p + 6, col[0]] = 1.0
        W[6*n + 0, col[0]] = 1.0
        U[1*m + j, col[0]] = 1.0
        W[6*n + 1, col[0]] = 1.0
        col[0] += 1
    # That's 4 terms, but we need 7. Let me just use 7 direct terms
    # Actually, let's use 8 direct terms and accept rank 50, then try to reduce
    
    # Fallback: use 8 direct terms
    while col[0] < 49:
        i = (col[0] - 42) % 2
        j = (col[0] - 42) // 2
        if j >= m:
            break
        U[i*m + j, col[0]] = 1.0
        V[j*p + 6, col[0]] = 1.0
        W[6*n + i, col[0]] = 1.0
        col[0] += 1
    
    if col[0] != 49:
        return None
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = 2, 4, 7
    
    # Try rank-49 reduction first
    if _time_left() > 20:
        result = _try_reduce_rank50(n, m, p)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    # Try direct rank-49 construction
    if _time_left() > 10:
        result = _direct_construction(n, m, p)
        if result is not None:
            if _verify(result["U"], result["V"], result["W"], n, m, p):
                return result
    
    # Fallback to rank-50
    return _construct_rank50(n, m, p)
