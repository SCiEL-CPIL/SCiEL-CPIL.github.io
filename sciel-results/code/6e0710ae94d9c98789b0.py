"""Combined program for 3x3 x 3x4 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 4]

def _verify_decomposition(U, V, W, n=3, m=3, p=4):
    """Verify rank-r decomposition: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1."""
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n=3, m=3, p=4):
    """Schoolbook rank n*m*p decomposition."""
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

def _add_schoolbook_block(U, V, W, n, m, p, i0, j0, k0, start_col):
    """Add schoolbook decomposition for sub-block M[n,m,p] at offsets i0,j0,k0."""
    col = start_col
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[(i0 + i) * m + (j0 + j)][col] = 1
                V[(j0 + j) * p + (k0 + k)][col] = 1
                W[(k0 + k) * n + (i0 + i)][col] = 1
                col += 1
    return col

def _add_strassen_2x2x2(U, V, W, n, m, p, i0, j0, k0, start_col):
    """
    Add Strassen decomposition for 2x2x2 block at offsets i0,j0,k0.
    Uses 7 rank terms.
    
    For the 2x2 block, we have:
    A = [[A00, A01], [A10, A11]]
    B = [[B00, B01], [B10, B11]]
    C = A*B = [[C00, C01], [C10, C11]]
    
    Strassen's 7 products:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    col = start_col
    
    # Index helpers
    def u_idx(ii, jj):
        return (i0 + ii) * m + (j0 + jj)
    
    def v_idx(jj, kk):
        return (j0 + jj) * p + (k0 + kk)
    
    def w_idx(kk, ii):
        return (k0 + kk) * n + (i0 + ii)
    
    # M1 = (A00+A11)(B00+B11)
    U[u_idx(0, 0)][col] = 1      # A00
    U[u_idx(1, 1)][col] = 1      # A11
    V[v_idx(0, 0)][col] = 1      # B00
    V[v_idx(1, 1)][col] = 1      # B11
    W[w_idx(0, 0)][col] = 1      # C00
    W[w_idx(1, 1)][col] = 1      # C11
    col += 1
    
    # M2 = (A10+A11)B00
    U[u_idx(1, 0)][col] = 1      # A10
    U[u_idx(1, 1)][col] = 1      # A11
    V[v_idx(0, 0)][col] = 1      # B00
    W[w_idx(0, 1)][col] = 1      # C10
    col += 1
    
    # M3 = A00(B10-B11)
    U[u_idx(0, 0)][col] = 1      # A00
    V[v_idx(1, 0)][col] = 1      # B10
    V[v_idx(1, 1)][col] = -1     # B11
    W[w_idx(1, 0)][col] = 1      # C01
    col += 1
    
    # M4 = A11(B01-B00)
    U[u_idx(1, 1)][col] = 1      # A11
    V[v_idx(0, 1)][col] = 1      # B01
    V[v_idx(0, 0)][col] = -1     # B00
    W[w_idx(0, 0)][col] = 1      # C00
    W[w_idx(0, 1)][col] = 1      # C10
    col += 1
    
    # M5 = (A00+A01)B11
    U[u_idx(0, 0)][col] = 1      # A00
    U[u_idx(0, 1)][col] = 1      # A01
    V[v_idx(1, 1)][col] = 1      # B11
    W[w_idx(0, 0)][col] = -1     # C00
    W[w_idx(1, 0)][col] = 1      # C01
    col += 1
    
    # M6 = (A10-A00)(B00+B01)
    U[u_idx(1, 0)][col] = 1      # A10
    U[u_idx(0, 0)][col] = -1     # A00
    V[v_idx(0, 0)][col] = 1      # B00
    V[v_idx(0, 1)][col] = 1      # B01
    W[w_idx(1, 1)][col] = 1      # C11
    col += 1
    
    # M7 = (A01-A11)(B10+B11)
    U[u_idx(0, 1)][col] = 1      # A01
    U[u_idx(1, 1)][col] = -1     # A11
    V[v_idx(1, 0)][col] = 1      # B10
    V[v_idx(1, 1)][col] = 1      # B11
    W[w_idx(0, 0)][col] = 1      # C00
    col += 1
    
    return col

def _build_rank34():
    """Build the rank-34 decomposition using Strassen for 2x2x2 blocks."""
    n, m, p = 3, 3, 4
    rank = 34
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # This is 2x2 x 2x2, use Strassen (rank 7)
    col = _add_strassen_2x2x2(U, V, W, n, m, p, 0, 0, 0, col)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # This is 2x2 x 2x2, use Strassen (rank 7)
    col = _add_strassen_2x2x2(U, V, W, n, m, p, 0, 0, 2, col)
    
    # Block 3: A[0:2, 2:3] * B[2:3, 0:4] -> C[0:2, 0:4]
    # This is 2x1 x 1x4, use schoolbook (rank 8)
    col = _add_schoolbook_block(U, V, W, 2, 1, 4, 0, 2, 0, col)
    
    # Block 4: A[2:3, 0:2] * B[0:2, 0:4] -> C[2:3, 0:4]
    # This is 1x2 x 2x4, use schoolbook (rank 8)
    col = _add_schoolbook_block(U, V, W, 1, 2, 4, 2, 0, 0, col)
    
    # Block 5: A[2:3, 2:3] * B[2:3, 0:4] -> C[2:3, 0:4]
    # This is 1x1 x 1x4, use schoolbook (rank 4)
    col = _add_schoolbook_block(U, V, W, 1, 1, 4, 2, 2, 0, col)
    
    if col != rank:
        print(f"Warning: Rank mismatch. Expected {rank}, got {col}")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = DIMENSIONS
    
    # Try the rank-34 construction first
    result = _build_rank34()
    
    # Verify
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # If rank-34 fails, try to find a working decomposition via local search
    # Start with schoolbook and try to reduce rank by merging columns
    
    # Fallback: return schoolbook (rank 36)
    return _schoolbook(n, m, p)
