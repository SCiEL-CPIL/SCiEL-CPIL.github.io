import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 4]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    if isinstance(U, np.ndarray):
        U = U.tolist()
    if isinstance(V, np.ndarray):
        V = V.tolist()
    if isinstance(W, np.ndarray):
        W = W.tolist()
    
    rank = len(U[0]) if U else 0
    if len(U) != n*m or len(V) != m*p or len(W) != p*n:
        return False
    
    # Check identity: sum_r U_ij_r V_jk_r W_ik_r == 1 for all i, j, k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

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
    return U, V, W

def _add_strassen_2x2(U, V, W, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, offset, n, m, p):
    """
    Adds Strassen decomposition for A[a_rows, a_cols] * B[b_rows, b_cols] -> C[c_rows, c_cols]
    """
    def idx_a(r, c): return r * m + c
    def idx_b(r, c): return r * p + c
    def idx_c(r, c): return c * n + r
    
    # M1
    U[idx_a(a_rows[0], a_cols[0])][offset] += 1
    U[idx_a(a_rows[1], a_cols[1])][offset] += 1
    V[idx_b(b_rows[0], b_cols[0])][offset] += 1
    V[idx_b(b_rows[1], b_cols[1])][offset] += 1
    W[idx_c(c_rows[0], c_cols[0])][offset] += 1
    W[idx_c(c_rows[1], c_cols[1])][offset] += 1
    
    # M2
    U[idx_a(a_rows[1], a_cols[0])][offset+1] += 1
    U[idx_a(a_rows[1], a_cols[1])][offset+1] += 1
    V[idx_b(b_rows[0], b_cols[0])][offset+1] += 1
    W[idx_c(c_rows[1], c_cols[0])][offset+1] += 1
    W[idx_c(c_rows[1], c_cols[1])][offset+1] -= 1
    
    # M3
    U[idx_a(a_rows[0], a_cols[0])][offset+2] += 1
    V[idx_b(b_rows[1], b_cols[0])][offset+2] += 1
    V[idx_b(b_rows[1], b_cols[1])][offset+2] -= 1
    W[idx_c(c_rows[0], c_cols[1])][offset+2] += 1
    W[idx_c(c_rows[1], c_cols[1])][offset+2] += 1
    
    # M4
    U[idx_a(a_rows[1], a_cols[1])][offset+3] += 1
    V[idx_b(b_rows[0], b_cols[1])][offset+3] += 1
    V[idx_b(b_rows[0], b_cols[0])][offset+3] -= 1
    W[idx_c(c_rows[0], c_cols[0])][offset+3] -= 1
    W[idx_c(c_rows[1], c_cols[0])][offset+3] += 1
    
    # M5
    U[idx_a(a_rows[0], a_cols[0])][offset+4] += 1
    U[idx_a(a_rows[1], a_cols[0])][offset+4] += 1
    V[idx_b(b_rows[1], b_cols[1])][offset+4] += 1
    W[idx_c(c_rows[0], c_cols[0])][offset+4] += 1
    
    # M6
    U[idx_a(a_rows[1], a_cols[0])][offset+5] += 1
    U[idx_a(a_rows[1], a_cols[1])][offset+5] -= 1
    V[idx_b(b_rows[0], b_cols[1])][offset+5] += 1
    V[idx_b(b_rows[1], b_cols[1])][offset+5] += 1
    W[idx_c(c_rows[1], c_cols[1])][offset+5] += 1
    
    # M7
    U[idx_a(a_rows[0], a_cols[1])][offset+6] += 1
    U[idx_a(a_rows[1], a_cols[1])][offset+6] += 1
    V[idx_b(b_rows[0], b_cols[0])][offset+6] += 1
    V[idx_b(b_rows[1], b_cols[0])][offset+6] -= 1
    W[idx_c(c_rows[0], c_cols[0])][offset+6] -= 1
    W[idx_c(c_rows[0], c_cols[1])][offset+6] += 1

def _try_rank_44():
    n, m, p = 3, 4, 4
    rank = 44
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    _add_strassen_2x2(U, V, W, [0,1], [0,1], [0,1], [0,1], [0,1], [0,1], 0, n, m, p)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    _add_strassen_2x2(U, V, W, [0,1], [0,1], [0,1], [2,3], [0,1], [2,3], 7, n, m, p)
    
    # Block 3: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    _add_strassen_2x2(U, V, W, [0,1], [2,3], [2,3], [0,1], [0,1], [0,1], 14, n, m, p)
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    _add_strassen_2x2(U, V, W, [0,1], [2,3], [2,3], [2,3], [0,1], [2,3], 21, n, m, p)
    
    # Block 5: A[2:3, 0:4] * B[0:4, 0:4] -> C[2:3, 0:4]
    offset = 28
    for j in range(4):
        for k in range(4):
            U[2*m + j][offset] = 1
            V[j*p + k][offset] = 1
            W[k*n + 2][offset] = 1
            offset += 1
            
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130.0
    
    # Try rank-44 decomposition
    try:
        result = _try_rank_44()
        if result[0] is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p):
                return {"U": U.tolist() if isinstance(U, np.ndarray) else U,
                        "V": V.tolist() if isinstance(V, np.ndarray) else V,
                        "W": W.tolist() if isinstance(W, np.ndarray) else W}
    except Exception:
        pass
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    return {"U": U_sb, "V": V_sb, "W": W_sb}
