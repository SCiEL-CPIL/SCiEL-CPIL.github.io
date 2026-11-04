import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank estimate: 16 (Part 1) + 4 * 7 (Part 2) = 44
    rank = 44
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: Row 0 of A (1x4) times B (4x4)
    # C[0, k] = sum_j A[0, j] B[j, k]
    # U index: 0*m + j = j
    # V index: j*p + k
    # W index: k*n + 0 = 3k
    for j in range(m):
        for k in range(p):
            U[j][col] = 1
            V[j * p + k][col] = 1
            W[k * n][col] = 1
            col += 1
            
    # Part 2: Rows 1,2 of A (2x4) times B (4x4)
    # Split A into A_L (cols 0,1) and A_R (cols 2,3)
    # Split B into B_TL, B_TR, B_BL, B_BR (2x2 blocks)
    
    # Helper to add a 2x2x2 Strassen decomposition
    # a_indices: list of 4 U indices for the 2x2 A block
    # b_indices: list of 4 V indices for the 2x2 B block
    # w_indices: list of 4 W indices for the 2x2 C block (row-major: c11, c12, c21, c22)
    # Note: W indices correspond to specific k, i pairs.
    
    def add_strassen(a_idx, b_idx, w_idx):
        nonlocal col
        # a_idx = [a11, a12, a21, a22]
        # b_idx = [b11, b12, b21, b22]
        # w_idx = [c11, c12, c21, c22]
        
        # M1 = (a11 + a22)(b11 + b22)
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[3]][col] += 1
        W[w_idx[0]][col] += 1 # c11
        W[w_idx[3]][col] += 1 # c22
        col += 1
        
        # M2 = (a21 + a22)b11
        U[a_idx[2]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[w_idx[2]][col] += 1 # c21
        W[w_idx[3]][col] -= 1 # c22
        col += 1
        
        # M3 = (a11 - a21)(b21 + b22)
        U[a_idx[0]][col] += 1
        U[a_idx[2]][col] -= 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] += 1
        W[w_idx[0]][col] -= 1 # c11
        W[w_idx[2]][col] -= 1 # c21
        col += 1
        
        # M4 = a11(b12 - b22)
        U[a_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        V[b_idx[3]][col] -= 1
        W[w_idx[1]][col] += 1 # c12
        W[w_idx[3]][col] += 1 # c22
        col += 1
        
        # M5 = (a12 + a22)(b11 + b21)
        U[a_idx[1]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[2]][col] += 1
        W[w_idx[0]][col] -= 1 # c11
        W[w_idx[1]][col] += 1 # c12
        col += 1
        
        # M6 = (a11 - a22)(b12 + b22)
        U[a_idx[0]][col] += 1
        U[a_idx[3]][col] -= 1
        V[b_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[w_idx[3]][col] += 1 # c22
        col += 1
        
        # M7 = (a21 - a11)(b11 - b12)
        U[a_idx[2]][col] += 1
        U[a_idx[0]][col] -= 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] -= 1
        W[w_idx[0]][col] -= 1 # c11
        W[w_idx[2]][col] += 1 # c21
        col += 1

    # Define indices for A_L, A_R
    # A_L is rows 1,2 cols 0,1
    # U indices: row 1 col 0: 1*4+0=4, row 1 col 1: 5, row 2 col 0: 8, row 2 col 1: 9
    a_L = [4, 5, 8, 9]
    # A_R is rows 1,2 cols 2,3
    # U indices: row 1 col 2: 6, row 1 col 3: 7, row 2 col 2: 10, row 2 col 3: 11
    a_R = [6, 7, 10, 11]
    
    # Define indices for B blocks
    # B_TL: rows 0,1 cols 0,1
    # V indices: r0c0=0, r0c1=1, r1c0=4, r1c1=5
    b_TL = [0, 1, 4, 5]
    # B_TR: rows 0,1 cols 2,3
    # V indices: r0c2=2, r0c3=3, r1c2=6, r1c3=7
    b_TR = [2, 3, 6, 7]
    # B_BL: rows 2,3 cols 0,1
    # V indices: r2c0=8, r2c1=9, r3c0=12, r3c1=13
    b_BL = [8, 9, 12, 13]
    # B_BR: rows 2,3 cols 2,3
    # V indices: r2c2=10, r2c3=11, r3c2=14, r3c3=15
    b_BR = [10, 11, 14, 15]
    
    # Define W indices for C blocks
    # C_TL corresponds to i=1,2 and k=0,1
    # c11 (i=1,k=0): 0*3+1 = 1
    # c12 (i=1,k=1): 1*3+1 = 4
    # c21 (i=2,k=0): 0*3+2 = 2
    # c22 (i=2,k=1): 1*3+2 = 5
    w_TL = [1, 4, 2, 5]
    
    # C_TR corresponds to i=1,2 and k=2,3
    # c11 (i=1,k=2): 2*3+1 = 7
    # c12 (i=1,k=3): 3*3+1 = 10
    # c21 (i=2,k=2): 2*3+2 = 8
    # c22 (i=2,k=3): 3*3+2 = 11
    w_TR = [7, 10, 8, 11]
    
    # P1 = A_L * B_TL -> contributes to C_TL
    add_strassen(a_L, b_TL, w_TL)
    
    # P2 = A_R * B_BL -> contributes to C_TL
    add_strassen(a_R, b_BL, w_TL)
    
    # P3 = A_L * B_TR -> contributes to C_TR
    add_strassen(a_L, b_TR, w_TR)
    
    # P4 = A_R * B_BR -> contributes to C_TR
    add_strassen(a_R, b_BR, w_TR)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
