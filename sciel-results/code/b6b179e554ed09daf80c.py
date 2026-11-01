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
    Returns U, V, W of shapes (4, 7), (4, 7), (4, 7).
    Convention:
      U indexed by (i, j) with row index i*m + j, m=2
      V indexed by (j, k) with row index j*p + k, p=2
      W indexed by (k, i) with row index k*n + i, n=2
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

def _build_strassen_2x2x11():
    """
    Build a rank-39 decomposition for 2x2 by 2x11 matrix multiplication.
    Uses 5 Strassen blocks of 2 columns each (covering columns 0-9),
    plus 1 schoolbook term for column 10.
    Total rank: 5*7 + 1 = 36... wait, that's 36.
    Actually: 5 Strassen blocks * 7 = 35, plus 1 for last column = 36.
    But we need to handle the 11th column. Let me use 5 full Strassen blocks (cols 0-9)
    and 1 schoolbook for col 10.
    Total rank = 5*7 + 1 = 36.
    
    Wait, the task says rank 39 was achieved by another program. Let me think...
    Maybe they used a different approach. Let me just build a valid decomposition.
    
    Actually, let me reconsider: 2x2 by 2x11.
    We can group the 11 columns into pairs: (0,1), (2,3), (4,5), (6,7), (8,9), and single (10).
    Each pair uses 7 Strassen terms, the single uses 1 schoolbook term.
    Total: 5*7 + 1 = 36.
    """
    n, m, p = 2, 2, 11
    total_rank = 36
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    U_base, V_base, W_base = _build_strassen_2x2x2()
    
    # Place 5 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7), (8,9)
    for block_idx in range(5):
        col_offset = block_idx * 2
        rank_offset = block_idx * 7
        
        # U is the same for all blocks (only depends on i, j)
        U[:, rank_offset:rank_offset+7] = U_base
        
        # V: map local k (0,1) to global k (col_offset, col_offset+1)
        for j in range(2):
            for k_local in range(2):
                v_row_global = j * p + (col_offset + k_local)
                v_row_base = j * 2 + k_local
                V[v_row_global, rank_offset:rank_offset+7] = V_base[v_row_base, :]
        
        # W: map local k (0,1) to global k (col_offset, col_offset+1)
        for k_local in range(2):
            for i in range(2):
                w_row_global = (col_offset + k_local) * n + i
                w_row_base = k_local * 2 + i
                W[w_row_global, rank_offset:rank_offset+7] = W_base[w_row_base, :]
    
    # Term for column 10: schoolbook
    # C[i, 10] = A[i, 0]*B[0, 10] + A[i, 1]*B[1, 10]
    # We need two terms for this, not one. Let me fix.
    # Actually, for a single column, we need m=2 terms.
    # So total rank should be 5*7 + 2 = 37.
    
    # Let me redo with correct count.
    total_rank = 5*7 + 2  # 37
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    for block_idx in range(5):
        col_offset = block_idx * 2
        rank_offset = block_idx * 7
        
        U[:, rank_offset:rank_offset+7] = U_base
        
        for j in range(2):
            for k_local in range(2):
                v_row_global = j * p + (col_offset + k_local)
                v_row_base = j * 2 + k_local
                V[v_row_global, rank_offset:rank_offset+7] = V_base[v_row_base, :]
        
        for k_local in range(2):
            for i in range(2):
                w_row_global = (col_offset + k_local) * n + i
                w_row_base = k_local * 2 + i
                W[w_row_global, rank_offset:rank_offset+7] = W_base[w_row_base, :]
    
    # Column 10: two schoolbook terms
    rank_offset = 5*7  # 35
    
    # Term 35: A[i,0] * B[0,10]
    # U[i,0,35] = 1, U[i,1,35] = 0
    # V[0,10,35] = 1, V[1,10,35] = 0
    # W[10,i,35] = 1
    for i in range(2):
        U[i*2+0, rank_offset] = 1.0  # i, j=0
    V[0*p+10, rank_offset] = 1.0  # j=0, k=10
    for i in range(2):
        W[10*n+i, rank_offset] = 1.0
    
    # Term 36: A[i,1] * B[1,10]
    rank_offset2 = rank_offset + 1
    for i in range(2):
        U[i*2+1, rank_offset2] = 1.0  # i, j=1
    V[1*p+10, rank_offset2] = 1.0  # j=1, k=10
    for i in range(2):
        W[10*n+i, rank_offset2] = 1.0
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 11
    deadline = time.time() + 115  # safety margin
    
    # Build rank-37 decomposition using 5 Strassen blocks + 2 schoolbook terms
    U, V, W = _build_strassen_2x2x11()
    
    # Verify
    if time.time() < deadline:
        if not _verify(U, V, W, n, m, p):
            # Fall back to schoolbook
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
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
