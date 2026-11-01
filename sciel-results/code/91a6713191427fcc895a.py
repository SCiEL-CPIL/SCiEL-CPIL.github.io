import numpy as np
import time

DIMENSIONS = [2, 2, 11]

def _verify(U, V, W, n, m, p, tol=1e-12):
    """Verify that the decomposition exactly matches the matrix multiplication tensor."""
    rank = U.shape[1]
    np.random.seed(42)
    for _ in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        C_dec = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                val = 0.0
                for j in range(m):
                    for c in range(rank):
                        val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                C_dec[i, k] = val
        
        if not np.allclose(C, C_dec, atol=tol):
            return False
    return True

def _build_strassen_2x2x2():
    """
    Build a rank-7 decomposition for 2x2 by 2x2 matrix multiplication.
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1: (a00 + a11)(b00 + b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2: (a10 + a11)b00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    # M3: a00(b01 - b11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    # M4: a11(b10 - b00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    # M5: (a00 + a01)b11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    # M6: (a10 - a00)(b00 + b10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[3, 5] = 1
    
    # M7: (a01 - a11)(b11 + b10)
    U[1, 6] = 1; U[3, 6] = -1
    V[3, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1; W[2, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 11
    deadline = time.time() + 115
    
    # Strategy: Use Strassen for 5 blocks of 2 columns, and Schoolbook for 1 block of 1 column.
    # Total rank: 5 * 7 + 1 * 4 = 39.
    
    U_base, V_base, W_base = _build_strassen_2x2x2()
    
    total_rank = 39
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    # 5 Strassen blocks
    for _ in range(5):
        # U
        U[:, col_offset:col_offset+7] = U_base
        
        # V
        for j in range(2):
            for k_local in range(2):
                global_k = col_offset + k_local
                v_row_global = j * p + global_k
                v_row_base = j * 2 + k_local
                V[v_row_global, col_offset:col_offset+7] = V_base[v_row_base, :]
        
        # W
        for k_local in range(2):
            for i in range(2):
                global_k = col_offset + k_local
                w_row_global = global_k * n + i
                w_row_base = k_local * 2 + i
                W[w_row_global, col_offset:col_offset+7] = W_base[w_row_base, :]
        
        col_offset += 7
    
    # 1 Schoolbook block (column 10)
    # Rank 4
    # U_sub: (4, 4)
    # V_sub: (2, 4)
    # W_sub: (2, 4)
    # Terms: (i=0,j=0), (i=0,j=1), (i=1,j=0), (i=1,j=1)
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    
    k_global = 10
    for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        c = col_offset + idx
        U[i*m+j, c] = 1
        V[j*p+k_global, c] = 1
        W[k_global*n+i, c] = 1
    
    # Verify
    if time.time() < deadline:
        if not _verify(U, V, W, n, m, p):
            # Fallback to schoolbook
            rank = n * m * p
            U_fb = np.zeros((n*m, rank))
            V_fb = np.zeros((m*p, rank))
            W_fb = np.zeros((p*n, rank))
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i*m+j, column] = 1
                        V_fb[j*p+k, column] = 1
                        W_fb[k*n+i, column] = 1
                        column += 1
            U, V, W = U_fb, V_fb, W_fb
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
