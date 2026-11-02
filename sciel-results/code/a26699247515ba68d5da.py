import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Rank is 28 (4 blocks of Strassen 2x2x2)
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block
    # a_rows, a_cols: indices of A submatrix
    # b_rows, b_cols: indices of B submatrix
    # c_rows, c_cols: indices of C submatrix
    # col_offset: starting column in U, V, W
    def add_strassen_block(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, col_offset):
        # Strassen variables for 2x2 matrices
        # M1 = (x11+x22)(y11+y22)
        # M2 = (x21+x22)y11
        # M3 = x11(y12-y22)
        # M4 = x22(y21-y11)
        # M5 = (x11+x12)y22
        # M6 = (x21-x11)(y11+y12)
        # M7 = (x12-x22)(y21+y22)
        
        # U (A) coefficients
        # x11 is A[a_rows[0], a_cols[0]]
        # x12 is A[a_rows[0], a_cols[1]]
        # x21 is A[a_rows[1], a_cols[0]]
        # x22 is A[a_rows[1], a_cols[1]]
        
        # M1
        U[a_rows[0] * m + a_cols[0]][col_offset] += 1
        U[a_rows[1] * m + a_cols[1]][col_offset] += 1
        V[b_rows[0] * p + b_cols[0]][col_offset] += 1
        V[b_rows[1] * p + b_cols[1]][col_offset] += 1
        # Z11 += M1, Z22 += M1
        W[c_cols[0] * n + c_rows[0]][col_offset] += 1
        W[c_cols[1] * n + c_rows[1]][col_offset] += 1
        
        # M2
        U[a_rows[1] * m + a_cols[0]][col_offset+1] += 1
        U[a_rows[1] * m + a_cols[1]][col_offset+1] += 1
        V[b_rows[0] * p + b_cols[0]][col_offset+1] += 1
        # Z21 += M2, Z22 -= M2
        W[c_cols[0] * n + c_rows[1]][col_offset+1] += 1
        W[c_cols[1] * n + c_rows[1]][col_offset+1] -= 1
        
        # M3
        U[a_rows[0] * m + a_cols[0]][col_offset+2] += 1
        V[b_rows[0] * p + b_cols[1]][col_offset+2] += 1
        V[b_rows[1] * p + b_cols[1]][col_offset+2] -= 1
        # Z12 += M3, Z22 += M3
        W[c_cols[1] * n + c_rows[0]][col_offset+2] += 1
        W[c_cols[1] * n + c_rows[1]][col_offset+2] += 1
        
        # M4
        U[a_rows[1] * m + a_cols[1]][col_offset+3] += 1
        V[b_rows[1] * p + b_cols[0]][col_offset+3] += 1
        V[b_rows[0] * p + b_cols[0]][col_offset+3] -= 1
        # Z11 += M4, Z21 += M4
        W[c_cols[0] * n + c_rows[0]][col_offset+3] += 1
        W[c_cols[0] * n + c_rows[1]][col_offset+3] += 1
        
        # M5
        U[a_rows[0] * m + a_cols[0]][col_offset+4] += 1
        U[a_rows[0] * m + a_cols[1]][col_offset+4] += 1
        V[b_rows[1] * p + b_cols[1]][col_offset+4] += 1
        # Z11 -= M5, Z12 += M5
        W[c_cols[0] * n + c_rows[0]][col_offset+4] -= 1
        W[c_cols[1] * n + c_rows[0]][col_offset+4] += 1
        
        # M6
        U[a_rows[1] * m + a_cols[0]][col_offset+5] += 1
        U[a_rows[0] * m + a_cols[0]][col_offset+5] -= 1
        V[b_rows[0] * p + b_cols[0]][col_offset+5] += 1
        V[b_rows[0] * p + b_cols[1]][col_offset+5] += 1
        # Z22 += M6
        W[c_cols[1] * n + c_rows[1]][col_offset+5] += 1
        
        # M7
        U[a_rows[0] * m + a_cols[1]][col_offset+6] += 1
        U[a_rows[1] * m + a_cols[1]][col_offset+6] -= 1
        V[b_rows[1] * p + b_cols[0]][col_offset+6] += 1
        V[b_rows[1] * p + b_cols[1]][col_offset+6] += 1
        # Z11 += M7
        W[c_cols[0] * n + c_rows[0]][col_offset+6] += 1

    # Block 1: A1 (rows 0,1; cols 0,1) * B1 (rows 0,1; cols 0,1) -> C (rows 0,1; cols 0,1)
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], 0)
    
    # Block 2: A1 (rows 0,1; cols 0,1) * B2 (rows 0,1; cols 2,3) -> C (rows 0,1; cols 2,3)
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3], 7)
    
    # Block 3: A2 (rows 0,1; cols 2,3) * B3 (rows 2,3; cols 0,1) -> C (rows 0,1; cols 0,1)
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1], 14)
    
    # Block 4: A2 (rows 0,1; cols 2,3) * B4 (rows 2,3; cols 2,3) -> C (rows 0,1; cols 2,3)
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3], 21)

    return {"U": U, "V": V, "W": W}
