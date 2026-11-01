import numpy as np

def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm coefficients for 2x2 multiplication
    # We have 4 blocks to process:
    # Block 0: A_L (cols 0,1) * B_LL (rows 0,1, cols 0,1) -> C_L (cols 0,1)
    # Block 1: A_R (cols 2,3) * B_RL (rows 2,3, cols 0,1) -> C_L (cols 0,1)
    # Block 2: A_L (cols 0,1) * B_LR (rows 0,1, cols 2,3) -> C_R (cols 2,3)
    # Block 3: A_R (cols 2,3) * B_RR (rows 2,3, cols 2,3) -> C_R (cols 2,3)
    
    # Helper function to add a Strassen block to the global matrices
    def add_block(block_idx, a_cols, b_rows, b_cols, c_cols):
        # a_cols: list of 2 column indices in A (global)
        # b_rows: list of 2 row indices in B (global)
        # b_cols: list of 2 column indices in B (global)
        # c_cols: list of 2 column indices in C (global)
        
        base = block_idx * 7
        
        # Local indices for 2x2 blocks
        # A is 2x2 local: rows 0,1 are fixed global rows 0,1 (since n=2)
        # B is 2x2 local: rows/cols mapped from b_rows/b_cols
        # C is 2x2 local: rows 0,1 fixed, cols mapped from c_cols
        
        # Strassen M1..M7 definitions
        # M1: (A00+A11)(B00+B11)
        # M2: (A10+A11)B00
        # M3: A00(B01-B11)
        # M4: A11(B10-B00)
        # M5: (A00+A01)B11
        # M6: (A10-A00)(B00+B01)
        # M7: (A01-A11)(B10+B11)
        
        # U updates (A part)
        # A00 is global row 0, col a_cols[0]
        # A01 is global row 0, col a_cols[1]
        # A10 is global row 1, col a_cols[0]
        # A11 is global row 1, col a_cols[1]
        
        # M1
        U[0 * m + a_cols[0]][base + 0] += 1
        U[1 * m + a_cols[1]][base + 0] += 1
        # M2
        U[1 * m + a_cols[0]][base + 1] += 1
        U[1 * m + a_cols[1]][base + 1] += 1
        # M3
        U[0 * m + a_cols[0]][base + 2] += 1
        # M4
        U[1 * m + a_cols[1]][base + 3] += 1
        # M5
        U[0 * m + a_cols[0]][base + 4] += 1
        U[0 * m + a_cols[1]][base + 4] += 1
        # M6
        U[1 * m + a_cols[0]][base + 5] += 1
        U[0 * m + a_cols[0]][base + 5] -= 1
        # M7
        U[0 * m + a_cols[1]][base + 6] += 1
        U[1 * m + a_cols[1]][base + 6] -= 1
        
        # V updates (B part)
        # B00 is global row b_rows[0], col b_cols[0]
        # B01 is global row b_rows[0], col b_cols[1]
        # B10 is global row b_rows[1], col b_cols[0]
        # B11 is global row b_rows[1], col b_cols[1]
        
        # M1
        V[b_rows[0] * p + b_cols[0]][base + 0] += 1
        V[b_rows[1] * p + b_cols[1]][base + 0] += 1
        # M2
        V[b_rows[0] * p + b_cols[0]][base + 1] += 1
        # M3
        V[b_rows[0] * p + b_cols[1]][base + 2] += 1
        V[b_rows[1] * p + b_cols[1]][base + 2] -= 1
        # M4
        V[b_rows[1] * p + b_cols[0]][base + 3] += 1
        V[b_rows[0] * p + b_cols[0]][base + 3] -= 1
        # M5
        V[b_rows[1] * p + b_cols[1]][base + 4] += 1
        # M6
        V[b_rows[0] * p + b_cols[0]][base + 5] += 1
        V[b_rows[0] * p + b_cols[1]][base + 5] += 1
        # M7
        V[b_rows[1] * p + b_cols[0]][base + 6] += 1
        V[b_rows[1] * p + b_cols[1]][base + 6] += 1
        
        # W updates (C part)
        # C00 is global row 0, col c_cols[0] -> W index c_cols[0]*n + 0
        # C01 is global row 0, col c_cols[1] -> W index c_cols[1]*n + 0
        # C10 is global row 1, col c_cols[0] -> W index c_cols[0]*n + 1
        # C11 is global row 1, col c_cols[1] -> W index c_cols[1]*n + 1
        
        # Z00 = M1 + M4 - M5 + M7
        W[c_cols[0] * n + 0][base + 0] += 1
        W[c_cols[0] * n + 0][base + 3] += 1
        W[c_cols[0] * n + 0][base + 4] -= 1
        W[c_cols[0] * n + 0][base + 6] += 1
        
        # Z01 = M3 + M5
        W[c_cols[1] * n + 0][base + 2] += 1
        W[c_cols[1] * n + 0][base + 4] += 1
        
        # Z10 = M2 + M4
        W[c_cols[0] * n + 1][base + 1] += 1
        W[c_cols[0] * n + 1][base + 3] += 1
        
        # Z11 = M1 - M2 + M3 + M6
        W[c_cols[1] * n + 1][base + 0] += 1
        W[c_cols[1] * n + 1][base + 1] -= 1
        W[c_cols[1] * n + 1][base + 2] += 1
        W[c_cols[1] * n + 1][base + 5] += 1

    # Define the 4 blocks
    # Block 0: A_L (cols 0,1), B_LL (rows 0,1, cols 0,1), C_L (cols 0,1)
    add_block(0, [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 1: A_R (cols 2,3), B_RL (rows 2,3, cols 0,1), C_L (cols 0,1)
    add_block(1, [2, 3], [2, 3], [0, 1], [0, 1])
    
    # Block 2: A_L (cols 0,1), B_LR (rows 0,1, cols 2,3), C_R (cols 2,3)
    add_block(2, [0, 1], [0, 1], [2, 3], [2, 3])
    
    # Block 3: A_R (cols 2,3), B_RR (rows 2,3, cols 2,3), C_R (cols 2,3)
    add_block(3, [2, 3], [2, 3], [2, 3], [2, 3])

    return {"U": U, "V": V, "W": W}
