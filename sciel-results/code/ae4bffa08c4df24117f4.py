"""Program for matrix multiplication tensor rank minimization (3x4 by 4x6)."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    # U: (n*m, rank) = (12, rank)
    # V: (m*p, rank) = (24, rank)
    # W: (p*n, rank) = (18, rank)
    
    # We will use a block decomposition.
    # Split A (3x4) into A_top (2x4) and A_bot (1x4).
    # Split B (4x6) into B_1 (4x2), B_2 (4x2), B_3 (4x2).
    # For each block b in {1,2,3}:
    #   C_b = A * B_b (3x2)
    #   Split C_b into C_top (2x2) and C_bot (1x2).
    #   C_top = A_top * B_b (2x2).
    #   C_bot = A_bot * B_b (1x2).
    
    # For C_top (2x2 multiplication):
    #   Use Strassen's algorithm (rank 7).
    #   A_top is 2x4. B_b is 4x2.
    #   Wait, A_top is 2x4, not 2x2.
    #   A_top = [A_top_1 (2x2) | A_top_2 (2x2)].
    #   B_b = [B_b_1 (2x2); B_b_2 (2x2)] (splitting rows).
    #   C_top = A_top_1 B_b_1 + A_top_2 B_b_2.
    #   This is sum of two 2x2x2 multiplications.
    #   Each 2x2x2 multiplication has rank 7 (Strassen).
    #   So rank for C_top is 7 + 7 = 14.
    
    # For C_bot (1x2 multiplication):
    #   A_bot is 1x4. B_b is 4x2.
    #   C_bot = A_bot * B_b.
    #   This is a dot product of 4 elements with each column of B_b.
    #   C_bot[0,0] = sum_j A_bot[j] B_b[j,0]
    #   C_bot[0,1] = sum_j A_bot[j] B_b[j,1]
    #   This can be done with 4 scalar multiplications?
    #   No, rank of 1x4x2 matrix multiplication.
    #   T_{i,j,k} = A_i B_{j,k}. Here i=0, j in 0..3, k in 0..1.
    #   Actually, A is 1x4, B is 4x2, C is 1x2.
    #   C_k = sum_j A_j B_{j,k}.
    #   This is just 4 multiplications for each k?
    #   Rank is 4*2 = 8?
    #   Let's check.
    #   C_0 = A_0 B_{0,0} + A_1 B_{1,0} + A_2 B_{2,0} + A_3 B_{3,0}
    #   C_1 = A_0 B_{0,1} + A_1 B_{1,1} + A_2 B_{2,1} + A_3 B_{3,1}
    #   We can compute these with 8 scalar multiplications:
    #   m_1 = A_0 B_{0,0}, m_2 = A_1 B_{1,0}, m_3 = A_2 B_{2,0}, m_4 = A_3 B_{3,0}
    #   m_5 = A_0 B_{0,1}, m_6 = A_1 B_{1,1}, m_7 = A_2 B_{2,1}, m_8 = A_3 B_{3,1}
    #   C_0 = m_1+m_2+m_3+m_4, C_1 = m_5+m_6+m_7+m_8.
    #   So rank is 8.
    
    # Total rank per block b: 14 + 8 = 22.
    # Total rank for 3 blocks: 3 * 22 = 66.
    
    # Let's construct the matrices.
    # U indices: (i, j) where i in 0..2, j in 0..3.
    # V indices: (j, k) where j in 0..3, k in 0..5.
    # W indices: (k, i) where k in 0..5, i in 0..2.
    
    # Strassen 2x2x2 factors:
    # U_strassen (4x7) for A (2x2):
    # u1 = a11
    # u2 = a11+a22
    # u3 = a22
    # u4 = a11-a22
    # u5 = a12+a22
    # u6 = a11+a12
    # u7 = a21-a22
    # V_strassen (4x7) for B (2x2):
    # v1 = b22
    # v2 = b11
    # v3 = b11+b12
    # v4 = b21-b22
    # v5 = b11+b21
    # v6 = b22-b12
    # v7 = b12
    # W_strassen (4x7) for C (2x2):
    # w11 = u1*v1 + u2*v2 + u3*v3 - u4*v4 - u5*v5
    # w12 = u5*v6 + u6*v7
    # w21 = u6*v1 + u7*v5
    # w22 = u1*v3 - u2*v2 + u3*v4 + u7*v5
    # Wait, standard Strassen:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Let's use this standard Strassen.
    # For A_top_1 (2x2):
    # A_top_1 = A[0:2, 0:2]
    # For B_b_1 (2x2):
    # B_b_1 = B[0:2, 2*(b-1) : 2*b]
    
    # We will build U, V, W as lists of lists.
    # rank = 66
    # We will fill them column by column.
    
    # Helper to get Strassen factors for a specific 2x2 block
    # A_block: 2x2, B_block: 2x2
    # Returns lists of (u_idx, val), (v_idx, val), (w_idx, val)
    
    # Let's just hardcode the 7 terms for each 2x2 multiplication.
    
    # Term 1: m1 = (a11+a22)(b11+b22)
    # Term 2: m2 = (a21+a22)b11
    # Term 3: m3 = a11(b12-b22)
    # Term 4: m4 = a22(b21-b11)
    # Term 5: m5 = (a11+a12)b22
    # Term 6: m6 = (a21-a11)(b11+b12)
    # Term 7: m7 = (a12-a22)(b21+b22)
    
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # For each block b (0, 1, 2):
    #   B_b cols: 2b, 2b+1
    #   B_b_1 rows: 0, 1. B_b_2 rows: 2, 3.
    #   A_top_1: rows 0,1; cols 0,1.
    #   A_top_2: rows 0,1; cols 2,3.
    #   A_bot: row 2; cols 0,1,2,3.
    
    # We will create a list of tuples for U, V, W contributions.
    # U_contribs: list of (u_row, val)
    # V_contribs: list of (v_row, val)
    # W_contribs: list of (w_row, val)
    
    # Since we are summing, we can just add to the matrices.
    
    U = [[0.0] * 66 for _ in range(12)]
    V = [[0.0] * 66 for _ in range(24)]
    W = [[0.0] * 66 for _ in range(18)]
    
    col = 0
    
    # Strassen terms for a 2x2 multiplication
    # A is 2x2, B is 2x2
    # A indices: (0,0), (0,1), (1,0), (1,1)
    # B indices: (0,0), (0,1), (1,0), (1,1)
    
    # We will define a function to add a Strassen multiplication
    # a_rows: [r1, r2] (global row indices in U for A)
    # a_cols: [c1, c2] (global col indices in A for A)
    # b_rows: [r1, r2] (global row indices in V for B)
    # b_cols: [c1, c2] (global col indices in V for B)
    # w_rows: [r1, r2] (global row indices in W for C)
    # w_cols: [c1, c2] (global col indices in C)
    # Note: U index is i*m+j. V index is j*p+k. W index is k*n+i.
    
    def add_strassen(a_rows, a_cols, b_rows, b_cols, w_rows, w_cols, col_start):
        nonlocal col
        # a_rows are global i indices. a_cols are global j indices.
        # b_rows are global j indices. b_cols are global k indices.
        # w_rows are global k indices. w_cols are global i indices.
        
        # A11, A12, A21, A22
        a11 = (a_rows[0], a_cols[0])
        a12 = (a_rows[0], a_cols[1])
        a21 = (a_rows[1], a_cols[0])
        a22 = (a_rows[1], a_cols[1])
        
        # B11, B12, B21, B22
        b11 = (b_rows[0], b_cols[0])
        b12 = (b_rows[0], b_cols[1])
        b21 = (b_rows[1], b_cols[0])
        b22 = (b_rows[1], b_cols[1])
        
        # C11, C12, C21, C22
        c11 = (w_rows[0], w_cols[0])
        c12 = (w_rows[0], w_cols[1])
        c21 = (w_rows[1], w_cols[0])
        c22 = (w_rows[1], w_cols[1])
        
        # m1 = (a11+a22)(b11+b22)
        U[a11[0]*4+a11[1]][col_start+0] += 1
        U[a22[0]*4+a22[1]][col_start+0] += 1
        V[b11[0]*6+b11[1]][col_start+0] += 1
        V[b22[0]*6+b22[1]][col_start+0] += 1
        W[c11[0]*3+c11[1]][col_start+0] += 1
        W[c22[0]*3+c22[1]][col_start+0] += 1
        
        # m2 = (a21+a22)b11
        U[a21[0]*4+a21[1]][col_start+1] += 1
        U[a22[0]*4+a22[1]][col_start+1] += 1
        V[b11[0]*6+b11[1]][col_start+1] += 1
        W[c21[0]*3+c21[1]][col_start+1] += 1
        W[c22[0]*3+c22[1]][col_start+1] -= 1
        
        # m3 = a11(b12-b22)
        U[a11[0]*4+a11[1]][col_start+2] += 1
        V[b12[0]*6+b12[1]][col_start+2] += 1
        V[b22[0]*6+b22[1]][col_start+2] -= 1
        W[c12[0]*3+c12[1]][col_start+2] += 1
        W[c22[0]*3+c22[1]][col_start+2] += 1
        
        # m4 = a22(b21-b11)
        U[a22[0]*4+a22[1]][col_start+3] += 1
        V[b21[0]*6+b21[1]][col_start+3] += 1
        V[b11[0]*6+b11[1]][col_start+3] -= 1
        W[c11[0]*3+c11[1]][col_start+3] += 1
        W[c21[0]*3+c21[1]][col_start+3] += 1
        
        # m5 = (a11+a12)b22
        U[a11[0]*4+a11[1]][col_start+4] += 1
        U[a12[0]*4+a12[1]][col_start+4] += 1
        V[b22[0]*6+b22[1]][col_start+4] += 1
        W[c11[0]*3+c11[1]][col_start+4] -= 1
        W[c12[0]*3+c12[1]][col_start+4] += 1
        
        # m6 = (a21-a11)(b11+b12)
        U[a21[0]*4+a21[1]][col_start+5] += 1
        U[a11[0]*4+a11[1]][col_start+5] -= 1
        V[b11[0]*6+b11[1]][col_start+5] += 1
        V[b12[0]*6+b12[1]][col_start+5] += 1
        W[c22[0]*3+c22[1]][col_start+5] += 1
        
        # m7 = (a12-a22)(b21+b22)
        U[a12[0]*4+a12[1]][col_start+6] += 1
        U[a22[0]*4+a22[1]][col_start+6] -= 1
        V[b21[0]*6+b21[1]][col_start+6] += 1
        V[b22[0]*6+b22[1]][col_start+6] += 1
        W[c11[0]*3+c11[1]][col_start+6] += 1
        
        return col_start + 7
    
    # For each block b in 0..2:
    for b in range(3):
        # C_top = A_top * B_b
        # A_top rows: 0,1. A_top cols: 0,1,2,3.
        # B_b rows: 0,1,2,3. B_b cols: 2b, 2b+1.
        
        # C_top = A_top_1 B_b_1 + A_top_2 B_b_2
        # A_top_1: rows 0,1; cols 0,1.
        # B_b_1: rows 0,1; cols 2b, 2b+1.
        # C_top_1: rows 2b, 2b+1; cols 0,1. (Wait, W index is k*n+i)
        # C_top is 2x2. C_top[k', i] where k' in 0,1 (local), i in 0,1 (global).
        # Global k for B_b is 2b + k'.
        
        # Term 1: A_top_1 * B_b_1
        a_rows = [0, 1]
        a_cols = [0, 1]
        b_rows = [0, 1]
        b_cols = [2*b, 2*b+1]
        w_rows = [2*b, 2*b+1]
        w_cols = [0, 1]
        col = add_strassen(a_rows, a_cols, b_rows, b_cols, w_rows, w_cols, col)
        
        # Term 2: A_top_2 * B_b_2
        a_rows = [0, 1]
        a_cols = [2, 3]
        b_rows = [2, 3]
        b_cols = [2*b, 2*b+1]
        w_rows = [2*b, 2*b+1]
        w_cols = [0, 1]
        col = add_strassen(a_rows, a_cols, b_rows, b_cols, w_rows, w_cols, col)
        
        # C_bot = A_bot * B_b
        # A_bot: row 2; cols 0,1,2,3.
        # B_b: rows 0,1,2,3; cols 2b, 2b+1.
        # C_bot: row 2b, 2b+1; col 2.
        
        # m_jk = A_bot[j] B_b[j, k]
        for j in range(4):
            for k_local in range(2):
                k_global = 2*b + k_local
                U[2*4 + j][col] += 1
                V[j*6 + k_global][col] += 1
                W[k_global*3 + 2][col] += 1
                col += 1
    
    # Convert to integers if possible
    for i in range(12):
        for j in range(66):
            if U[i][j] == int(U[i][j]):
                U[i][j] = int(U[i][j])
    for i in range(24):
        for j in range(66):
            if V[i][j] == int(V[i][j]):
                V[i][j] = int(V[i][j])
    for i in range(18):
        for j in range(66):
            if W[i][j] == int(W[i][j]):
                W[i][j] = int(W[i][j])
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
