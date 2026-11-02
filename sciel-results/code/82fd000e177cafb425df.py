import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(block_id, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols):
        base_col = block_id * 7
        
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

    # Block 1: A_L * B_LL -> C_L
    add_strassen_block(0, [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A_R * B_RL -> C_L
    add_strassen_block(1, [0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block 3: A_L * B_LR -> C_R
    add_strassen_block(2, [0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block 4: A_R * B_RR -> C_R
    add_strassen_block(3, [0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])

    return {"U": U, "V": V, "W": W}
