import numpy as np

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 30
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # U_S: 4x7, V_S: 4x7, W_S: 4x7
    U_S = [
        [1, 0, 1, 0, 1, -1, 0], # A11
        [0, 0, 0, 0, 1, 0, 1],  # A12
        [0, 1, 0, 0, 0, 1, 0],  # A21
        [1, 1, 0, 1, 0, 0, -1]  # A22
    ]
    V_S = [
        [1, 1, 0, -1, 0, 1, 0], # B11
        [0, 0, 1, 0, 0, 1, 0],  # B12
        [0, 0, 0, 1, 0, 0, 1],  # B21
        [1, 0, -1, 0, 1, 0, 1]  # B22
    ]
    W_S = [
        [1, 0, 0, 1, -1, 0, 1], # C11
        [0, 0, 1, 0, 1, 0, 0],  # C12
        [0, 1, 0, 1, 0, 0, 0],  # C21
        [1, -1, 1, 0, 0, 1, 0]  # C22
    ]
    
    # Helper to add a Strassen block
    # a_rows: indices in U for A_sub (4 rows)
    # b_rows: indices in V for B_sub (4 rows)
    # c_rows: indices in W for C_sub (4 rows)
    # col_offset: starting column in U, V, W
    def add_strassen(a_rows, b_rows, c_rows, col_offset):
        for r in range(7):
            for i in range(4):
                U[a_rows[i]][col_offset + r] += U_S[i][r]
                V[b_rows[i]][col_offset + r] += V_S[i][r]
                W[c_rows[i]][col_offset + r] += W_S[i][r]
                
    # Helper to add outer product
    # a_rows: indices in U (2 rows)
    # b_rows: indices in V (4 rows)
    # c_rows: indices in W (8 rows)
    # col_offset: starting column
    def add_outer(a_rows, b_rows, c_rows, col_offset):
        for i in range(2):
            U[a_rows[i]][col_offset] = 1
        for k in range(4):
            V[b_rows[k]][col_offset] = 1
        for i in range(2):
            for k in range(4):
                W[c_rows[k*2+i]][col_offset] = 1
                
    # Block 1 (cols 0-14)
    # A_left: rows 0,1,3,4 in U
    # B1_top_L: rows 0,1,8,9 in V (j=0,1; k=0,1)
    # C1_L: rows 0,2,1,3 in W (k=0,1; i=0,1)
    add_strassen([0,1,3,4], [0,1,8,9], [0,2,1,3], 0)
    
    # B1_top_R: rows 2,3,10,11 in V (j=0,1; k=2,3)
    # C1_R: rows 4,6,5,7 in W (k=2,3; i=0,1)
    add_strassen([0,1,3,4], [2,3,10,11], [4,6,5,7], 7)
    
    # a_right: rows 2,5 in U
    # b1_bottom: rows 16,17,18,19 in V (j=2; k=0,1,2,3)
    # C1_bottom: rows 0..7 in W
    add_outer([2,5], [16,17,18,19], [0,1,2,3,4,5,6,7], 14)
    
    # Block 2 (cols 15-29)
    # B2_top_L: rows 4,5,12,13 in V (j=0,1; k=4,5)
    # C2_L: rows 8,10,9,11 in W (k=4,5; i=0,1)
    add_strassen([0,1,3,4], [4,5,12,13], [8,10,9,11], 15)
    
    # B2_top_R: rows 6,7,14,15 in V (j=0,1; k=6,7)
    # C2_R: rows 12,14,13,15 in W (k=6,7; i=0,1)
    add_strassen([0,1,3,4], [6,7,14,15], [12,14,13,15], 22)
    
    # b2_bottom: rows 20,21,22,23 in V (j=2; k=4,5,6,7)
    # C2_bottom: rows 8..15 in W
    add_outer([2,5], [20,21,22,23], [8,9,10,11,12,13,14,15], 29)
    
    return {"U": U, "V": V, "W": W}
