import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition of 6x6 matrix multiplication using a block-Strassen approach.
    
    The 6x6 matrices are partitioned into 3x3 blocks of 2x2 submatrices.
    The multiplication is performed by:
    1. Iterating over block row I, block col J, and block col K (3x3x3 = 27 block products).
    2. For each block product A[I,J] * B[J,K], using the rank-7 Strassen decomposition.
    
    Total rank = 27 * 7 = 189.
    """
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Strassen factors for 2x2 matrix multiplication
    # U_s: 4x7, V_s: 4x7, W_s: 4x7
    # Indices for 2x2: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # U_s[i, s] is the coefficient of A[i] in the s-th Strassen product
    # V_s[j, s] is the coefficient of B[j] in the s-th Strassen product
    # W_s[k, s] is the coefficient for C[k] in the s-th Strassen product
    
    # Strassen formulas:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A10)B11
    # M6 = (A01 - A00)(B00 + B10)
    # M7 = (A10 - A11)(B01 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Initialize Strassen factors (4x7)
    Us = np.zeros((4, 7), dtype=int)
    Vs = np.zeros((4, 7), dtype=int)
    Ws = np.zeros((4, 7), dtype=int)
    
    # s=0 (M1)
    Us[0, 0] = 1; Us[3, 0] = 1  # A00, A11
    Vs[0, 0] = 1; Vs[3, 0] = 1  # B00, B11
    Ws[0, 0] = 1; Ws[3, 0] = 1  # C00, C11
    
    # s=1 (M2)
    Us[2, 1] = 1; Us[3, 1] = 1  # A10, A11
    Vs[0, 1] = 1                # B00
    Ws[2, 1] = 1; Ws[3, 1] = -1 # C10, C11
    
    # s=2 (M3)
    Us[0, 2] = 1                # A00
    Vs[1, 2] = 1; Vs[3, 2] = -1 # B01, B11
    Ws[1, 2] = 1; Ws[3, 2] = 1  # C01, C11
    
    # s=3 (M4)
    Us[3, 3] = 1                # A11
    Vs[2, 3] = 1; Vs[0, 3] = -1 # B10, B00
    Ws[0, 3] = 1; Ws[2, 3] = 1  # C00, C10
    
    # s=4 (M5)
    Us[0, 4] = 1; Us[2, 4] = 1  # A00, A10
    Vs[3, 4] = 1                # B11
    Ws[0, 4] = -1; Ws[1, 4] = 1 # C00, C01
    
    # s=5 (M6)
    Us[1, 5] = 1; Us[0, 5] = -1 # A01, A00
    Vs[0, 5] = 1; Vs[2, 5] = 1  # B00, B10
    Ws[3, 5] = 1                # C11
    
    # s=6 (M7)
    Us[2, 6] = 1; Us[3, 6] = -1 # A10, A11
    Vs[1, 6] = 1; Vs[3, 6] = 1  # B01, B11
    Ws[0, 6] = 1                # C00
    
    rank = 189
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for I in range(3):
        for J in range(3):
            for K in range(3):
                for s in range(7):
                    # U part: A block (I, J)
                    for a in range(2):
                        for b in range(2):
                            val = Us[a * 2 + b, s]
                            if val != 0:
                                i = 2 * I + a
                                j = 2 * J + b
                                U[i * m + j, col] = val
                    
                    # V part: B block (J, K)
                    for c in range(2):
                        for d in range(2):
                            val = Vs[c * 2 + d, s]
                            if val != 0:
                                j = 2 * J + c
                                k = 2 * K + d
                                V[j * p + k, col] = val
                    
                    # W part: C block (I, K)
                    for a in range(2):
                        for d in range(2):
                            val = Ws[a * 2 + d, s]
                            if val != 0:
                                i = 2 * I + a
                                k = 2 * K + d
                                W[k * n + i, col] = val
                    
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
