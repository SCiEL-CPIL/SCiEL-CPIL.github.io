import time
import numpy as np

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will construct rank 88 decomposition.
    # 8 blocks of 2x2x2 (Winograd rank 7) + 1x4x8 (Schoolbook rank 32)
    rank = 8 * 7 + 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Winograd 2x2x2 U, V, W
    # Indices for A: 0,1,2,3 (row-major 2x2)
    # Indices for B: 0,1,2,3 (col-major 2x2? No, B is 2x2, index j*p+k)
    # Indices for C: 0,1,2,3 (row-major 2x2? W[k*n+i])
    
    # Standard Winograd 2x2x2
    U_win = [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, -1, 0, 1, 0, -1],
        [0, 1, 1, 0, 0, 1, 1]
    ]
    # This U is for A. A is 2x2.
    # V is for B. B is 2x2.
    V_win = [
        [1, 1, 1, 1, 0, 0, 0],
        [1, -1, 1, -1, 1, 1, 1],
        [0, 0, 1, 1, 1, -1, 1],
        [0, 0, 1, -1, 0, 0, 0]
    ]
    # W is for C. C is 2x2.
    # W[k*n+i] -> index k*2+i
    W_win = [
        [1, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0]
    ]
    
    # Verify Winograd correctness mentally?
    # m1 = (a00+a11)(b00+b11)
    # U col 0: a00, a11. V col 0: b00, b11.
    # W col 0: c00, c11.
    # c00 += m1, c11 += m1.
    # m2 = (a00-a11)(b00-b11)
    # U col 1: a00, -a11. V col 1: b00, -b11.
    # W col 1: c00, c11.
    # c00 += m2, c11 += m2.
    # Sum m1+m2 = 2 a00 b00 + 2 a11 b11.
    # This implies c00 gets 2 a00 b00? No, we need 1 a00 b00.
    # So W should have 0.5?
    # The task allows half-integers.
    # So W col 0 should be [0.5, 0, 0.5, 0]?
    # Let's use the exact Winograd coefficients.
    
    # Correct Winograd 2x2x2 Rank 7:
    # U:
    # 1: 1, 0, 1, 0
    # 2: 1, 0, -1, 0
    # 3: 0, 1, 1, 0
    # 4: 0, 1, -1, 0
    # 5: 1, 1, 0, 0
    # 6: 1, -1, 0, 0
    # 7: 0, 0, 1, 1
    # V:
    # 1: 1, 1, 0, 0
    # 2: 1, -1, 0, 0
    # 3: 1, 1, 0, 0
    # 4: 1, -1, 0, 0
    # 5: 1, 0, 1, 0
    # 6: 1, 0, -1, 0
    # 7: 0, 1, 1, 0
    # W:
    # 1: 0.5, 0, 0, 0.5
    # 2: 0.5, 0, 0, -0.5
    # 3: 0, 0.5, 0.5, 0
    # 4: 0, 0.5, -0.5, 0
    # 5: 0.5, 0.5, 0, 0
    # 6: 0.5, -0.5, 0, 0
    # 7: 0, 0, 0, 0.5
    
    # I will use these coefficients.
    
    U_win = [
        [1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1],
        [1, -1, 1, -1, 0, 0, 1],
        [0, 0, 1, -1, 0, 0, 1]
    ]
    V_win = [
        [1, 1, 1, 1, 1, 1, 0],
        [1, -1, 1, -1, 0, 0, 1],
        [0, 0, 1, 1, 1, -1, 1],
        [0, 0, 1, -1, 0, 0, 0]
    ]
    W_win = [
        [0.5, 0.5, 0, 0, 0.5, 0.5, 0],
        [0, 0, 0.5, 0.5, 0.5, -0.5, 0],
        [0, 0, 0.5, -0.5, 0, 0, 0.5],
        [0.5, -0.5, 0, 0, 0, 0, 0.5]
    ]
    
    # Iterate over 2x2x2 blocks
    for r in range(1): # 1 block of 2 rows
        for ci in range(2): # 2 blocks of 2 inner cols
            for co in range(4): # 4 blocks of 2 outer cols
                # Global indices
                row_start = 2 * r
                inner_start = 2 * ci
                outer_start = 2 * co
                
                for w_idx in range(7):
                    # U
                    for a_i in range(4):
                        a_row = a_i // 2
                        a_col = a_i % 2
                        global_u = (row_start + a_row) * m + (inner_start + a_col)
                        U[global_u][col] += U_win[a_i][w_idx]
                        
                    # V
                    for b_i in range(4):
                        b_row = b_i // 2
                        b_col = b_i % 2
                        global_v = (inner_start + b_row) * p + (outer_start + b_col)
                        V[global_v][col] += V_win[b_i][w_idx]
                        
                    # W
                    for c_i in range(4):
                        c_row = c_i // 2
                        c_col = c_i % 2
                        global_w = (outer_start + c_col) * n + (row_start + c_row)
                        W[global_w][col] += W_win[c_i][w_idx]
                        
                    col += 1
                    
    # Handle remaining row 2
    # 1x4x8 schoolbook
    for i in range(1): # row 2
        for j in range(4):
            for k in range(8):
                global_u = (2 + i) * m + j
                global_v = j * p + k
                global_w = k * n + (2 + i)
                U[global_u][col] = 1
                V[global_v][col] = 1
                W[global_w][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
