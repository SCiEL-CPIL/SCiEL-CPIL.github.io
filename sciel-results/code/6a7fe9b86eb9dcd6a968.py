"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition of rank 22.
    # Strategy: Split p=4 into p1=2, p2=2.
    # Each part is 2x3x2 multiplication.
    # For 2x3x2, split m=3 into m1=2, m2=1.
    # Part 1: 2x2x2 (Strassen, rank 7)
    # Part 2: 2x1x2 (Rank 4)
    # Total for 2x3x2: 11.
    # Total for 2x3x4: 11 + 11 = 22.
    
    rank = 22
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    def add_strassen_222(u_offset, v_offset, w_offset, a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        # Adds a 2x2x2 Strassen decomposition.
        # a_rows: indices of A rows (0,1)
        # a_cols: indices of A cols (0,1)
        # b_rows: indices of B rows (0,1)
        # b_cols: indices of B cols (0,1)
        # c_rows: indices of C rows (0,1)
        # c_cols: indices of C cols (0,1)
        
        # Strassen factors for 2x2x2
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M7
        
        # We define U, V, W for each of the 7 terms.
        
        # Term 1
        U[a_rows[0] * m + a_cols[0]][col_offset] = 1
        U[a_rows[1] * m + a_cols[1]][col_offset] = 1
        V[b_rows[0] * p + b_cols[0]][col_offset] = 1
        V[b_rows[1] * p + b_cols[1]][col_offset] = 1
        W[c_cols[0] * n + c_rows[0]][col_offset] = 1
        W[c_cols[1] * n + c_rows[1]][col_offset] = 1
        col_offset += 1
        
        # Term 2
        U[a_rows[1] * m + a_cols[0]][col_offset] = 1
        U[a_rows[1] * m + a_cols[1]][col_offset] = 1
        V[b_rows[0] * p + b_cols[0]][col_offset] = 1
        W[c_cols[0] * n + c_rows[1]][col_offset] = 1
        W[c_cols[1] * n + c_rows[1]][col_offset] = -1
        col_offset += 1
        
        # Term 3
        U[a_rows[0] * m + a_cols[0]][col_offset] = 1
        V[b_rows[1] * p + b_cols[0]][col_offset] = 1
        V[b_rows[1] * p + b_cols[1]][col_offset] = -1
        W[c_cols[0] * n + c_rows[0]][col_offset] = 1
        W[c_cols[1] * n + c_rows[0]][col_offset] = 1
        col_offset += 1
        
        # Term 4
        U[a_rows[1] * m + a_cols[1]][col_offset] = 1
        V[b_rows[0] * p + b_cols[1]][col_offset] = 1
        V[b_rows[0] * p + b_cols[0]][col_offset] = -1
        W[c_cols[0] * n + c_rows[0]][col_offset] = 1
        W[c_cols[1] * n + c_rows[1]][col_offset] = 1
        col_offset += 1
        
        # Term 5
        U[a_rows[0] * m + a_cols[0]][col_offset] = 1
        U[a_rows[0] * m + a_cols[1]][col_offset] = 1
        V[b_rows[1] * p + b_cols[1]][col_offset] = 1
        W[c_cols[0] * n + c_rows[0]][col_offset] = -1
        W[c_cols[1] * n + c_rows[0]][col_offset] = 1
        col_offset += 1
        
        # Term 6
        U[a_rows[1] * m + a_cols[0]][col_offset] = 1
        U[a_rows[0] * m + a_cols[0]][col_offset] = -1
        V[b_rows[0] * p + b_cols[0]][col_offset] = 1
        V[b_rows[0] * p + b_cols[1]][col_offset] = 1
        W[c_cols[0] * n + c_rows[1]][col_offset] = 1
        col_offset += 1
        
        # Term 7
        U[a_rows[0] * m + a_cols[1]][col_offset] = 1
        U[a_rows[1] * m + a_cols[1]][col_offset] = -1
        V[b_rows[1] * p + b_cols[0]][col_offset] = 1
        V[b_rows[1] * p + b_cols[1]][col_offset] = 1
        W[c_cols[1] * n + c_rows[0]][col_offset] = -1
        W[c_cols[1] * n + c_rows[1]][col_offset] = 1
        col_offset += 1
        
        return col_offset

    def add_outer_212(u_offset, v_offset, w_offset, a_row, a_col, b_row, b_col, c_rows, c_cols):
        # Adds a 2x1x2 decomposition (rank 4).
        # A is 2x1 at (a_row, a_col)
        # B is 1x2 at (b_row, b_col)
        # C is 2x2 at (c_rows, c_cols)
        # C_{ik} = A_{i, a_col} B_{b_row, k}
        
        for i_idx in range(2):
            for k_idx in range(2):
                U[a_rows[i_idx] * m + a_col][col_offset] = 1
                V[b_row * p + b_col + k_idx][col_offset] = 1
                W[c_cols[k_idx] * n + c_rows[i_idx]][col_offset] = 1
                col_offset += 1
                
        return col_offset

    # We need to handle indices carefully.
    # A is 2x3. Rows 0,1. Cols 0,1,2.
    # B is 3x4. Rows 0,1,2. Cols 0,1,2,3.
    # C is 4x2. Rows 0,1,2,3. Cols 0,1.
    # W index: k*n + i. k is C row, i is C col.
    
    # Split p=4 into p1=2 (cols 0,1), p2=2 (cols 2,3).
    
    # Part 1: A (2x3) * B1 (3x2)
    # Split m=3 into m1=2 (cols 0,1 of A, rows 0,1 of B), m2=1 (col 2 of A, row 2 of B).
    
    # 1a: A11 (2x2) * B11 (2x2)
    # A rows 0,1. A cols 0,1.
    # B rows 0,1. B cols 0,1.
    # C rows 0,1. C cols 0,1.
    col_offset = add_strassen_222(
        a_rows=[0, 1], a_cols=[0, 1],
        b_rows=[0, 1], b_cols=[0, 1],
        c_rows=[0, 1], c_cols=[0, 1]
    )
    
    # 1b: A12 (2x1) * B12 (1x2)
    # A rows 0,1. A col 2.
    # B row 2. B cols 0,1.
    # C rows 0,1. C cols 0,1.
    col_offset = add_outer_212(
        a_rows=[0, 1], a_col=2,
        b_row=2, b_col=0,
        c_rows=[0, 1], c_cols=[0, 1]
    )
    
    # Part 2: A (2x3) * B2 (3x2)
    # B2 cols are 2,3.
    # Split m=3 into m1=2, m2=1.
    
    # 2a: A11 (2x2) * B21 (2x2)
    # A rows 0,1. A cols 0,1.
    # B rows 0,1. B cols 2,3.
    # C rows 2,3. C cols 0,1.
    col_offset = add_strassen_222(
        a_rows=[0, 1], a_cols=[0, 1],
        b_rows=[0, 1], b_cols=[2, 3],
        c_rows=[2, 3], c_cols=[0, 1]
    )
    
    # 2b: A12 (2x1) * B22 (1x2)
    # A rows 0,1. A col 2.
    # B row 2. B cols 2,3.
    # C rows 2,3. C cols 0,1.
    col_offset = add_outer_212(
        a_rows=[0, 1], a_col=2,
        b_row=2, b_col=2,
        c_rows=[2, 3], c_cols=[0, 1]
    )
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
