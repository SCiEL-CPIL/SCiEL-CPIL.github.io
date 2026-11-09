import time
import sys

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 80 decomposition
    # 3x4x4 part:
    # A_top (2x4) * B_2 (4x4) -> 2x4x4 rank 28 (4 * Strassen 2x2x2)
    # A_bot (1x4) * B_2 (4x4) -> 1x4x4 rank 16
    # 3x4x3 part:
    # A_top (2x4) * B_1 (4x3) -> 2x4x3 rank 24 (2 * 2x2x3 rank 12)
    # A_bot (1x4) * B_1 (4x3) -> 1x4x3 rank 12
    
    rank = 80
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 block
    # a_rows: list of 2 row indices in A
    # a_cols: list of 2 col indices in A
    # b_rows: list of 2 row indices in B
    # b_cols: list of 2 col indices in B
    # c_rows: list of 2 row indices in C (same as a_rows)
    # c_cols: list of 2 col indices in C (same as b_cols)
    def add_strassen_2x2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols):
        nonlocal col
        # M1 = (A11+A22)(B11+B22)
        U[a_rows[0]*m + a_cols[0]][col] = 1
        U[a_rows[1]*m + a_cols[1]][col] = 1
        V[b_rows[0]*p + b_cols[0]][col] = 1
        V[b_rows[1]*p + b_cols[1]][col] = 1
        W[c_cols[0]*n + c_rows[0]][col] = 1
        W[c_cols[1]*n + c_rows[0]][col] = 1
        W[c_cols[0]*n + c_rows[1]][col] = 1
        W[c_cols[1]*n + c_rows[1]][col] = 1
        # Adjust W for Strassen combination
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # This helper is too complex to do inline with single col.
        # We need to add 7 columns.
        pass

    # Let's just manually add the 7 columns for each 2x2x2 block.
    def add_strassen_block(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        # a_r: [r1, r2], a_c: [c1, c2]
        # b_r: [r1, r2], b_c: [c1, c2]
        # c_r: [r1, r2], c_c: [c1, c2]
        
        # M1
        U[a_r[0]*m + a_c[0]][col] = 1
        U[a_r[1]*m + a_c[1]][col] = 1
        V[b_r[0]*p + b_c[0]][col] = 1
        V[b_r[1]*p + b_c[1]][col] = 1
        W[c_c[0]*n + c_r[0]][col] = 1
        W[c_c[1]*n + c_r[0]][col] = 1
        W[c_c[0]*n + c_r[1]][col] = 1
        W[c_c[1]*n + c_r[1]][col] = 1
        col += 1
        
        # M2
        U[a_r[1]*m + a_c[0]][col] = 1
        U[a_r[1]*m + a_c[1]][col] = 1
        V[b_r[0]*p + b_c[0]][col] = 1
        W[c_c[0]*n + c_r[1]][col] = 1
        W[c_c[1]*n + c_r[1]][col] = 1
        col += 1
        
        # M3
        U[a_r[0]*m + a_c[0]][col] = 1
        V[b_r[0]*p + b_c[1]][col] = 1
        V[b_r[1]*p + b_c[1]][col] = -1
        W[c_c[1]*n + c_r[0]][col] = 1
        W[c_c[1]*n + c_r[1]][col] = 1
        col += 1
        
        # M4
        U[a_r[1]*m + a_c[1]][col] = 1
        V[b_r[1]*p + b_c[0]][col] = 1
        V[b_r[0]*p + b_c[0]][col] = -1
        W[c_c[0]*n + c_r[0]][col] = 1
        W[c_c[1]*n + c_r[0]][col] = 1
        W[c_c[0]*n + c_r[1]][col] = 1
        W[c_c[1]*n + c_r[1]][col] = 1
        col += 1
        
        # M5
        U[a_r[0]*m + a_c[0]][col] = 1
        U[a_r[0]*m + a_c[1]][col] = 1
        V[b_r[1]*p + b_c[1]][col] = 1
        W[c_c[0]*n + c_r[0]][col] = -1
        W[c_c[1]*n + c_r[0]][col] = -1
        W[c_c[0]*n + c_r[1]][col] = 1
        W[c_c[1]*n + c_r[1]][col] = 1
        col += 1
        
        # M6
        U[a_r[1]*m + a_c[0]][col] = 1
        U[a_r[0]*m + a_c[0]][col] = -1
        V[b_r[0]*p + b_c[0]][col] = 1
        V[b_r[0]*p + b_c[1]][col] = 1
        W[c_c[1]*n + c_r[1]][col] = 1
        W[c_c[1]*n + c_r[0]][col] = 1
        col += 1
        
        # M7
        U[a_r[0]*m + a_c[1]][col] = 1
        U[a_r[1]*m + a_c[1]][col] = -1
        V[b_r[1]*p + b_c[0]][col] = 1
        V[b_r[1]*p + b_c[1]][col] = 1
        W[c_c[0]*n + c_r[0]][col] = 1
        W[c_c[1]*n + c_r[0]][col] = 1
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    add_strassen_block([0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    add_strassen_block([0, 1], [2, 3], [2, 3], [0, 1], [0, 1], [0, 1])
    
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    add_strassen_block([0, 1], [0, 1], [0, 1], [2, 3], [0, 1], [2, 3])
    
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    add_strassen_block([0, 1], [2, 3], [2, 3], [2, 3], [0, 1], [2, 3])
    
    # Block 5: A[2:3, 0:4] * B[0:4, 0:4] -> C[2:3, 0:4]
    # 1x4x4 rank 16
    for j in range(4):
        for k in range(4):
            U[2*m + j][col] = 1
            V[j*p + k][col] = 1
            W[k*n + 2][col] = 1
            col += 1
            
    # Block 6: A[0:2, 0:2] * B[0:2, 4:7] -> C[0:2, 4:7]
    # 2x2x3 rank 12
    for j in range(2):
        for k in range(2):
            for c in range(3):
                U[j*m + k][col] = 1
                V[(j*2+k)*p + (4+c)][col] = 1
                W[(4+c)*n + j][col] = 1
                col += 1
                
    # Block 7: A[0:2, 2:4] * B[2:4, 4:7] -> C[0:2, 4:7]
    # 2x2x3 rank 12
    for j in range(2):
        for k in range(2):
            for c in range(3):
                U[j*m + (2+k)][col] = 1
                V[(2+j*2+k)*p + (4+c)][col] = 1
                W[(4+c)*n + j][col] = 1
                col += 1
                
    # Block 8: A[2:3, 0:4] * B[0:4, 4:7] -> C[2:3, 4:7]
    # 1x4x3 rank 12
    for j in range(4):
        for c in range(3):
            U[2*m + j][col] = 1
            V[j*p + (4+c)][col] = 1
            W[(4+c)*n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
