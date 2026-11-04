import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Block structure:
    # A (2x4) -> 2x2 blocks of 1x2
    # B (4x6) -> 2x2 blocks of 2x3
    # C (2x6) -> 2x2 blocks of 1x3
    
    # Strassen rank 7 for 2x2 block multiplication
    # Each block multiplication is 1x2 * 2x3, rank 6
    # Total rank 7 * 6 = 42
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Coefficients for Strassen steps s in 0..6
    # alpha[s][i][b] : coeff of A[i][b] in L_s
    # gamma[s][row][col] : coeff of B[row][col] in R_s
    # delta[i][col][s] : coeff of M_s in C[i][col]
    
    alpha = [
        [[1, 0], [0, 1]], # s=0: A00 + A11
        [[0, 0], [1, 1]], # s=1: A10 + A11
        [[1, 0], [0, 0]], # s=2: A00
        [[0, 0], [0, 1]], # s=3: A11
        [[1, 1], [0, 0]], # s=4: A00 + A01
        [[-1, 0], [1, 0]],# s=5: A10 - A00
        [[0, 1], [0, -1]] # s=6: A01 - A11
    ]
    
    gamma = [
        [[1, 0], [0, 1]], # s=0: B00 + B11
        [[1, 0], [0, 0]], # s=1: B00
        [[0, 1], [0, -1]],# s=2: B01 - B11
        [[-1, 0], [1, 0]],# s=3: B10 - B00
        [[0, 0], [0, 1]], # s=4: B11
        [[1, 1], [0, 0]], # s=5: B00 + B01
        [[0, 0], [1, 1]]  # s=6: B10 + B11
    ]
    
    delta = [
        # i=0
        [
            [1, 1, 0, 1, -1, 0, 1], # col=0: M0+M3-M4+M6 (indices 0,3,4,6)
            [0, 0, 1, 0, 1, 0, 0]   # col=1: M2+M4 (indices 2,4)
        ],
        # i=1
        [
            [0, 1, 0, 1, 0, 0, 0],  # col=0: M1+M3 (indices 1,3)
            [1, -1, 0, 0, 0, 1, 0]  # col=1: M0-M1+M5 (indices 0,1,5)
        ]
    ]
    
    col_idx = 0
    for s in range(7):
        for j_local in range(2): # 0, 1 (row in 2x3 block / col in 1x2 block)
            for k_prime in range(3): # 0, 1, 2 (col in 2x3 block / col in 1x3 block)
                # Fill U
                for i in range(2):
                    for b in range(2):
                        a_val = alpha[s][i][b]
                        if a_val != 0:
                            # A index: i * m + (b * 2 + j_local)
                            row_U = i * m + (b * 2 + j_local)
                            U[row_U][col_idx] = a_val
                            
                # Fill V
                for row_B in range(2):
                    for col_B in range(2):
                        g_val = gamma[s][row_B][col_B]
                        if g_val != 0:
                            # B index: (row_B * 2 + j_local) * p + (col_B * 3 + k_prime)
                            row_V = (row_B * 2 + j_local) * p + (col_B * 3 + k_prime)
                            V[row_V][col_idx] = g_val
                            
                # Fill W
                for i in range(2):
                    for col_C in range(2):
                        d_val = delta[i][col_C][s]
                        if d_val != 0:
                            # C index: (col_C * 3 + k_prime) * n + i
                            row_W = (col_C * 3 + k_prime) * n + i
                            W[row_W][col_idx] = d_val
                            
                col_idx += 1
                
    return {"U": U, "V": V, "W": W}
