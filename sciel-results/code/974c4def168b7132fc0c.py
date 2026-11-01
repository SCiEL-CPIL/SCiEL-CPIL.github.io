import numpy as np

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # M1 = (X11+X22)(Y11+Y22)
    # M2 = (X21+X22)Y11
    # M3 = X11(Y12-Y22)
    # M4 = X22(Y21-Y11)
    # M5 = (X11+X12)Y22
    # M6 = (X21-X11)(Y11+Y12)
    # M7 = (X12-X22)(Y21+Y22)
    
    def add_strassen_block(base_col, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        # A_rows, A_cols: 2 indices for A block
        # B_rows, B_cols: 2 indices for B block
        # C_rows, C_cols: 2 indices for C block
        
        # M1: (X11+X22)(Y11+Y22)
        col = base_col + 0
        U[A_rows[0] * m + A_cols[0]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        # M2: (X21+X22)Y11
        col = base_col + 1
        U[A_rows[1] * m + A_cols[0]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        
        # M3: X11(Y12-Y22)
        col = base_col + 2
        U[A_rows[0] * m + A_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] -= 1
        
        # M4: X22(Y21-Y11)
        col = base_col + 3
        U[A_rows[1] * m + A_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[0]][col] -= 1
        
        # M5: (X11+X12)Y22
        col = base_col + 4
        U[A_rows[0] * m + A_cols[0]][col] += 1
        U[A_rows[0] * m + A_cols[1]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        # M6: (X21-X11)(Y11+Y12)
        col = base_col + 5
        U[A_rows[1] * m + A_cols[0]][col] += 1
        U[A_rows[0] * m + A_cols[0]][col] -= 1
        V[B_rows[0] * p + B_cols[0]][col] += 1
        V[B_rows[0] * p + B_cols[1]][col] += 1
        
        # M7: (X12-X22)(Y21+Y22)
        col = base_col + 6
        U[A_rows[0] * m + A_cols[1]][col] += 1
        U[A_rows[1] * m + A_cols[1]][col] -= 1
        V[B_rows[1] * p + B_cols[0]][col] += 1
        V[B_rows[1] * p + B_cols[1]][col] += 1
        
        # W updates
        # Z11 = M1 + M4 - M5 + M7
        w_idx = C_cols[0] * n + C_rows[0]
        W[w_idx][base_col + 0] += 1
        W[w_idx][base_col + 3] += 1
        W[w_idx][base_col + 4] -= 1
        W[w_idx][base_col + 6] += 1
        
        # Z12 = M3 + M5
        w_idx = C_cols[1] * n + C_rows[0]
        W[w_idx][base_col + 2] += 1
        W[w_idx][base_col + 4] += 1
        
        # Z21 = M2 + M4
        w_idx = C_cols[0] * n + C_rows[1]
        W[w_idx][base_col + 1] += 1
        W[w_idx][base_col + 3] += 1
        
        # Z22 = M1 - M2 + M3 + M6
        w_idx = C_cols[1] * n + C_rows[1]
        W[w_idx][base_col + 0] += 1
        W[w_idx][base_col + 1] -= 1
        W[w_idx][base_col + 2] += 1
        W[w_idx][base_col + 5] += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block(7, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block(14, [0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block(21, [0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])

    return {"U": U, "V": V, "W": W}
