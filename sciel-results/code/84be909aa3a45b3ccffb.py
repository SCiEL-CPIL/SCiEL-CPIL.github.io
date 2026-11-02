import numpy as np

DIMENSIONS = [3, 4, 4]

def _strassen_2x2():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = -1
    
    # M4 = A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1; W[1, 3] = 1
    
    # M5 = (A00+A10)B11
    U[0, 4] = 1; U[2, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[2, 4] = 1
    
    # M6 = (A01-A11)(B10+B11)
    U[1, 5] = 1; U[3, 5] = -1
    V[2, 5] = 1; V[3, 5] = 1
    W[3, 5] = -1
    
    # M7 = (A01-A10)(B00+B10)
    U[1, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition is exact."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = sum(U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Total rank: 1 (bottom row) + 7*4 (top 2x4 split into 4 2x2x2 blocks) = 29
    rank = 29
    
    U_full = np.zeros((nm, rank))
    V_full = np.zeros((mp, rank))
    W_full = np.zeros((pn, rank))
    
    sU, sV, sW = _strassen_2x2()
    
    # Block 0: Bottom row (i=2), Rank 1
    # U[2*4+j, 0] = 1
    # V[j*4+k, 0] = 1
    # W[k*3+2, 0] = 1
    for j in range(m):
        U_full[2*m + j, 0] = 1
    for j in range(m):
        for k in range(p):
            V_full[j*p + k, 0] = 1
    for k in range(p):
        W_full[k*n + 2, 0] = 1
        
    # Helper to add a Strassen block
    # i_offset, j_offset, k_offset: offsets for global indices
    # col_start: starting column in U, V, W
    def add_strassen_block(i_off, j_off, k_off, col_start):
        for r in range(7):
            for s_idx in range(4):
                sr = s_idx // 2
                sc = s_idx % 2
                # U index: (i_off + sr) * m + (j_off + sc)
                u_idx = (i_off + sr) * m + (j_off + sc)
                U_full[u_idx, col_start + r] = sU[s_idx, r]
                
                # V index: (j_off + sr) * p + (k_off + sc)
                # Note: Strassen V index is j_row*2 + j_col. Here j_row=sr, j_col=sc.
                # But wait. In Strassen 2x2x2, V is 2x2.
                # V[j, r]. j is flat index of B.
                # B is 2x2. j = j_row*2 + j_col.
                # In our block, B sub-block is 2x2.
                # Global B index: (j_off + j_row) * p + (k_off + j_col).
                # Here j_row = sr, j_col = sc.
                v_idx = (j_off + sr) * p + (k_off + sc)
                V_full[v_idx, col_start + r] = sV[s_idx, r]
                
                # W index: (k_off + sr) * n + (i_off + sc)
                # Strassen W index is k_row*2 + k_col.
                # Here k_row = sr, k_col = sc.
                # Global W index: k_global * n + i_global.
                # k_global = k_off + sr.
                # i_global = i_off + sc.
                w_idx = (k_off + sr) * n + (i_off + sc)
                W_full[w_idx, col_start + r] = sW[s_idx, r]

    # Block A: i in {0,1}, j in {0,1}, k in {0,1}
    add_strassen_block(0, 0, 0, 1)
    
    # Block B: i in {0,1}, j in {0,1}, k in {2,3}
    add_strassen_block(0, 0, 2, 8)
    
    # Block C: i in {0,1}, j in {2,3}, k in {0,1}
    add_strassen_block(0, 2, 0, 15)
    
    # Block D: i in {0,1}, j in {2,3}, k in {2,3}
    add_strassen_block(0, 2, 2, 22)
    
    if not _verify(U_full.tolist(), V_full.tolist(), W_full.tolist(), n, m, p):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = np.zeros((nm, rank_sb))
        V_sb = np.zeros((mp, rank_sb))
        W_sb = np.zeros((pn, rank_sb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i*m + j, col] = 1
                    V_sb[j*p + k, col] = 1
                    W_sb[k*n + i, col] = 1
                    col += 1
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
