"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for 2x2x2
    # a_indices: indices in A (global U)
    # b_indices: indices in B (global V)
    # c_indices: indices in C (global W)
    def add_strassen_222(a_idx, b_idx, c_idx, start_col):
        # M1 = (A11+A22)(B11+B22)
        U[a_idx[0]][start_col] += 1
        U[a_idx[3]][start_col] += 1
        V[b_idx[0]][start_col] += 1
        V[b_idx[3]][start_col] += 1
        W[c_idx[0]][start_col] += 1
        W[c_idx[3]][start_col] += 1
        W[c_idx[1]][start_col] += 1
        W[c_idx[2]][start_col] += 1
        col_val = start_col + 1
        
        # M2 = (A21+A22)B11
        U[a_idx[2]][col_val] += 1
        U[a_idx[3]][col_val] += 1
        V[b_idx[0]][col_val] += 1
        W[c_idx[2]][col_val] += 1
        W[c_idx[3]][col_val] += 1
        col_val += 1
        
        # M3 = A11(B21-B22)
        U[a_idx[0]][col_val] += 1
        V[b_idx[2]][col_val] += 1
        V[b_idx[3]][col_val] -= 1
        W[c_idx[1]][col_val] += 1
        W[c_idx[3]][col_val] += 1
        col_val += 1
        
        # M4 = A22(B12-B11)
        U[a_idx[3]][col_val] += 1
        V[b_idx[1]][col_val] += 1
        V[b_idx[0]][col_val] -= 1
        W[c_idx[0]][col_val] += 1
        W[c_idx[2]][col_val] += 1
        col_val += 1
        
        # M5 = (A11+A12)B22
        U[a_idx[0]][col_val] += 1
        U[a_idx[1]][col_val] += 1
        V[b_idx[3]][col_val] += 1
        W[c_idx[0]][col_val] -= 1
        W[c_idx[1]][col_val] += 1
        col_val += 1
        
        # M6 = (A21-A11)(B11+B12)
        U[a_idx[2]][col_val] += 1
        U[a_idx[0]][col_val] -= 1
        V[b_idx[0]][col_val] += 1
        V[b_idx[1]][col_val] += 1
        W[c_idx[3]][col_val] += 1
        col_val += 1
        
        # M7 = (A12-A22)(B21+B22)
        U[a_idx[1]][col_val] += 1
        U[a_idx[3]][col_val] -= 1
        V[b_idx[2]][col_val] += 1
        V[b_idx[3]][col_val] += 1
        W[c_idx[0]][col_val] += 1
        W[c_idx[1]][col_val] -= 1
        col_val += 1
        
        return col_val

    # Helper to add schoolbook block for 2x2x1
    def add_schoolbook_221(a_idx, b_idx, c_idx, start_col):
        # C11 = A11 B11
        U[a_idx[0]][start_col] += 1
        V[b_idx[0]][start_col] += 1
        W[c_idx[0]][start_col] += 1
        col_val = start_col + 1
        
        # C12 = A12 B12
        U[a_idx[1]][col_val] += 1
        V[b_idx[1]][col_val] += 1
        W[c_idx[1]][col_val] += 1
        col_val += 1
        
        # C21 = A21 B21
        U[a_idx[2]][col_val] += 1
        V[b_idx[0]][col_val] += 1
        W[c_idx[2]][col_val] += 1
        col_val += 1
        
        # C22 = A22 B22
        U[a_idx[3]][col_val] += 1
        V[b_idx[1]][col_val] += 1
        W[c_idx[3]][col_val] += 1
        col_val += 1
        
        return col_val

    # Helper to add schoolbook block for 2x1x5
    def add_schoolbook_215(a_idx, b_idx, c_idx, start_col):
        # A is 2x1, B is 1x5
        # C[i,k] = A[i,0] B[0,k]
        for i in range(2):
            for k in range(5):
                U[a_idx[i]][start_col] += 1
                V[b_idx[k]][start_col] += 1
                W[c_idx[i][k]][start_col] += 1
                start_col += 1
        return start_col

    # Define indices
    # A is 2x3. U index = i*m + j.
    # A_L (2x2): i in 0,1; j in 0,1.
    a_L_idx = [0, 1, 2, 3] # A[0,0], A[0,1], A[1,0], A[1,1]
    # A_R (2x1): i in 0,1; j in 2.
    a_R_idx = [2, 4] # A[0,2], A[1,2]
    
    # B is 3x5. V index = j*p + k.
    # B_L (2x5): j in 0,1; k in 0..4.
    # B_L1 (2x2): k in 0,1.
    b_L1_idx = [0, 1, 5, 6] # B[0,0], B[0,1], B[1,0], B[1,1]
    # B_L2 (2x2): k in 2,3.
    b_L2_idx = [2, 3, 7, 8] # B[0,2], B[0,3], B[1,2], B[1,3]
    # B_L3 (2x1): k in 4.
    b_L3_idx = [4, 9] # B[0,4], B[1,4]
    # B_R (1x5): j in 2; k in 0..4.
    b_R_idx = [10, 11, 12, 13, 14] # B[2,0]..B[2,4]
    
    # C is 5x2 (W index = k*n + i).
    # C_L (5x2): k in 0..4; i in 0,1.
    # C_L1 (2x2 part of C_L? No, C_L is 2x5 result of A_L B_L.
    # A_L is 2x2, B_L is 2x5. Result is 2x5.
    # C_L1 corresponds to cols 0,1 of C_L.
    c_L1_idx = [0, 1, 2, 3] # C[0,0], C[0,1], C[1,0], C[1,1]
    # C_L2 corresponds to cols 2,3 of C_L.
    c_L2_idx = [2, 3, 4, 5] # C[0,2], C[0,3], C[1,2], C[1,3]
    # C_L3 corresponds to col 4 of C_L.
    c_L3_idx = [4, 6] # C[0,4], C[1,4]
    
    # C_R (5x2 part of C_R? A_R is 2x1, B_R is 1x5. Result is 2x5.
    # C_R corresponds to all cols of C.
    # c_R_idx[i][k] is index for C[i,k].
    c_R_idx = []
    for i in range(2):
        row = []
        for k in range(5):
            row.append(k * 2 + i)
        c_R_idx.append(row)

    # Execute blocks
    col = add_strassen_222(a_L_idx, b_L1_idx, c_L1_idx, col)
    col = add_strassen_222(a_L_idx, b_L2_idx, c_L2_idx, col)
    col = add_schoolbook_221(a_L_idx, b_L3_idx, c_L3_idx, col)
    col = add_schoolbook_215(a_R_idx, b_R_idx, c_R_idx, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
