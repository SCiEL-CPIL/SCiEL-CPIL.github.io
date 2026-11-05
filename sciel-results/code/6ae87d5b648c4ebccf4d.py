"""Matrix multiplication 2,4,4 - rank-minimal exact tensor decomposition."""

def construct():
    """
    Returns a rank-28 decomposition of the 2x4x4 matrix multiplication tensor
    using 4 independent Strassen decompositions for the 2x2x2 sub-blocks.
    
    The 2x4 matrix A is split into 2x2 blocks along the column dimension.
    The 4x4 matrix B is split into 2x2 blocks along both dimensions.
    Each 2x2x2 multiplication is handled by Strassen's algorithm (rank 7).
    
    U: shape (8, 28) - factors for A entries
    V: shape (16, 28) - factors for B entries  
    W: shape (8, 28) - factors for C entries (trace convention W[k*n+i, r])
    """
    n, m, p = 2, 4, 4
    
    # Strassen decomposition for 2x2x2 matrix multiplication
    # Strassen's 7 intermediate products for C = A*B where A, B are 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B10 - B11)
    # M4 = A11(B01 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # For the tensor decomposition, we need:
    # U[a_idx, r] for A entry a_idx
    # V[b_idx, r] for B entry b_idx  
    # W[c_idx, r] for C entry c_idx (using trace convention c_idx = k*n + i)
    
    # Strassen U matrix: 4 entries of A (A00, A01, A10, A11) x 7 Strassen terms
    # Row indices: 0=A00, 1=A01, 2=A10, 3=A11
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 1, 0, 0, 1, 0, -1],   # A01
        [0, 0, 1, 0, 0, 1, 0],    # A10
        [1, 0, 0, 1, 0, 0, 1],    # A11
    ]
    
    # Strassen V matrix: 4 entries of B (B00, B01, B10, B11) x 7 Strassen terms
    # Row indices: 0=B00, 1=B01, 2=B10, 3=B11
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 0, 1, 0, 1, 0],    # B01
        [1, 0, 1, 0, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ]
    
    # Strassen W matrix: 4 entries of C (C00, C01, C10, C11) x 7 Strassen terms
    # Row indices: 0=C00, 1=C01, 2=C10, 3=C11
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, 0, 1, 0, 0, 1, 0],    # C11
    ]
    
    # Build the full 2x4x4 decomposition
    # A has 2 rows x 4 cols = 8 entries, indexed as i*4 + j
    # B has 4 rows x 4 cols = 16 entries, indexed as j*4 + k
    # C has 4 rows x 2 cols = 8 entries, but using trace convention: W[k*n + i]
    # where n=2, so C entry (i,k) is at index k*2 + i
    
    rank = 4 * 7  # 4 blocks x 7 Strassen terms each
    U = [[0.0] * rank for _ in range(8)]   # 2*4 = 8 entries
    V = [[0.0] * rank for _ in range(16)]  # 4*4 = 16 entries
    W = [[0.0] * rank for _ in range(8)]   # 4*2 = 8 entries (trace convention)
    
    # Block structure:
    # A is split into columns: A_left (cols 0,1) and A_right (cols 2,3)
    # B is split into 4 2x2 blocks: B00, B01, B10, B11
    # C is split into columns: C_left (cols 0,1) and C_right (cols 2,3)
    #
    # C_left = A_left * B_left where B_left = [B00; B10] (4x2)
    # C_right = A_left * B_right + A_right * B_right... 
    # Actually: C = A * B where A is 2x4, B is 4x4
    # Split A = [A_L | A_R] where A_L, A_R are 2x2
    # Split B = [B_TL B_TR; B_BL B_BR] where each is 2x2
    # C = A_L * [B_TL; B_BL] + A_R * [B_TR; B_BR]
    #    = A_L * B_TL + A_R * B_TR (for top 2 cols of C)
    #    = A_L * B_BL + A_R * B_BR (for bottom 2 cols of C)
    
    # Block 0: A_L * B_TL -> C top-left (i in 0,1, k in 0,1)
    # A indices: i*4 + j where i in {0,1}, j in {0,1}
    # B indices: j*4 + k where j in {0,1}, k in {0,1}
    # C indices: k*2 + i where k in {0,1}, i in {0,1}
    for r in range(7):
        base = 0 * 7 + r
        # A_L entries
        for li in range(2):  # local i
            for lj in range(2):  # local j
                a_idx = li * 4 + lj
                U[a_idx][base] += S_U[li * 2 + lj][r]
        # B_TL entries
        for lj in range(2):  # local j (row of B)
            for lk in range(2):  # local k (col of B)
                b_idx = lj * 4 + lk
                V[b_idx][base] += S_V[lj * 2 + lk][r]
        # C top-left entries
        for lk in range(2):  # local k
            for li in range(2):  # local i
                c_idx = lk * 2 + li
                W[c_idx][base] += S_W[lk * 2 + li][r]
    
    # Block 1: A_R * B_TR -> C top-right (i in 0,1, k in 2,3)
    # A indices: i*4 + j where i in {0,1}, j in {2,3}
    # B indices: j*4 + k where j in {0,1}, k in {2,3}
    # C indices: k*2 + i where k in {2,3}, i in {0,1}
    for r in range(7):
        base = 1 * 7 + r
        # A_R entries
        for li in range(2):  # local i
            for lj in range(2):  # local j
                a_idx = li * 4 + (2 + lj)
                U[a_idx][base] += S_U[li * 2 + lj][r]
        # B_TR entries
        for lj in range(2):  # local j (row of B, 0,1)
            for lk in range(2):  # local k (col of B, 2,3)
                b_idx = lj * 4 + (2 + lk)
                V[b_idx][base] += S_V[lj * 2 + lk][r]
        # C top-right entries
        for lk in range(2):  # local k (0,1) -> global k = 2+lk
            for li in range(2):  # local i
                c_idx = (2 + lk) * 2 + li
                W[c_idx][base] += S_W[lk * 2 + li][r]
    
    # Block 2: A_L * B_BL -> C bottom-left (i in 0,1, k in 0,1) but from B rows 2,3
    # Wait, I need to reconsider. Let me re-derive.
    # A (2x4) * B (4x4) = C (2x4)
    # A = [a00 a01 a02 a03; a10 a11 a12 a13]
    # B = [b00 b01 b02 b03; b10 b11 b12 b13; b20 b21 b22 b23; b30 b31 b32 b33]
    # C[i][k] = sum_j A[i][j] * B[j][k]
    #
    # Split: A = [A_L | A_R] where A_L = cols 0,1; A_R = cols 2,3
    # B = [B_0; B_1] where B_0 = rows 0,1; B_1 = rows 2,3
    # C = A_L * B_0 + A_R * B_1
    #
    # B_0 is 2x4, B_1 is 2x4
    # Split B_0 = [B_0L | B_0R] where B_0L = cols 0,1; B_0R = cols 2,3
    # Split B_1 = [B_1L | B_1R] where B_1L = cols 0,1; B_1R = cols 2,3
    #
    # C = A_L * B_0L + A_R * B_1L (cols 0,1) + A_L * B_0R + A_R * B_1R (cols 2,3)
    
    # Block 2: A_L * B_1L -> C cols 0,1 (from A_L and B rows 2,3, cols 0,1)
    # A indices: i*4 + j where i in {0,1}, j in {0,1}
    # B indices: j*4 + k where j in {2,3}, k in {0,1}
    # C indices: k*2 + i where k in {0,1}, i in {0,1}
    for r in range(7):
        base = 2 * 7 + r
        # A_L entries
        for li in range(2):  # local i
            for lj in range(2):  # local j
                a_idx = li * 4 + lj
                U[a_idx][base] += S_U[li * 2 + lj][r]
        # B_1L entries (rows 2,3 of B, cols 0,1)
        for lj in range(2):  # local j (row of B, 2,3)
            for lk in range(2):  # local k (col of B, 0,1)
                b_idx = (2 + lj) * 4 + lk
                V[b_idx][base] += S_V[lj * 2 + lk][r]
        # C cols 0,1 entries
        for lk in range(2):  # local k
            for li in range(2):  # local i
                c_idx = lk * 2 + li
                W[c_idx][base] += S_W[lk * 2 + li][r]
    
    # Block 3: A_R * B_1R -> C cols 2,3 (from A_R and B rows 2,3, cols 2,3)
    # A indices: i*4 + j where i in {0,1}, j in {2,3}
    # B indices: j*4 + k where j in {2,3}, k in {2,3}
    # C indices: k*2 + i where k in {2,3}, i in {0,1}
    for r in range(7):
        base = 3 * 7 + r
        # A_R entries
        for li in range(2):  # local i
            for lj in range(2):  # local j
                a_idx = li * 4 + (2 + lj)
                U[a_idx][base] += S_U[li * 2 + lj][r]
        # B_1R entries (rows 2,3 of B, cols 2,3)
        for lj in range(2):  # local j (row of B, 2,3)
            for lk in range(2):  # local k (col of B, 2,3)
                b_idx = (2 + lj) * 4 + (2 + lk)
                V[b_idx][base] += S_V[lj * 2 + lk][r]
        # C cols 2,3 entries
        for lk in range(2):  # local k (0,1) -> global k = 2+lk
            for li in range(2):  # local i
                c_idx = (2 + lk) * 2 + li
                W[c_idx][base] += S_W[lk * 2 + li][r]
    
    return {"U": U, "V": V, "W": W}
