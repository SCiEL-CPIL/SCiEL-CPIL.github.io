import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a low-rank exact tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    
    Strategy: Hybrid Strassen/Schoolbook.
    Split A (3x3) into 2x2, 2x1, 1x2, 1x1 blocks.
    Split B (3x5) into 2x2, 2x3, 1x2, 1x3 blocks.
    
    C_TL = A_TL B_TL + A_TR B_BL
    C_TR = A_TL B_TR + A_TR B_BR
    C_BL = A_BL B_TL + A_BR B_BL
    C_BR = A_BL B_TR + A_BR B_BR
    
    Use Strassen for A_TL B_TL (rank 7).
    Use Schoolbook for all other terms.
    
    Total Rank = 7 (Strassen) + 4 (A_TR B_BL) + 12 (A_TL B_TR) + 6 (A_TR B_BR) 
                + 4 (A_BL B_TL) + 2 (A_BR B_BL) + 6 (A_BL B_TR) + 3 (A_BR B_BR)
                = 44.
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    start_time = time.time()
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 9 x 44
    V = [[0] * rank for _ in range(m * p)]  # 15 x 44
    W = [[0] * rank for _ in range(p * n)]  # 15 x 44
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i * m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1

    # Helper to add a Strassen term
    # Strassen for 2x2 * 2x2
    # M1 = A00 B00
    # M2 = (A00+A11) B11
    # M3 = A10 (B01-B11)
    # M4 = A11 (B00+B10)
    # M5 = (A00+A10) B10
    # M6 = (A01-A11) (B00+B10)
    # M7 = (A00-A11) (B11-B01)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Indices for A_TL (2x2) in A (3x3):
    # A_TL[0,0] is A[0,0] -> idx 0
    # A_TL[0,1] is A[0,1] -> idx 1
    # A_TL[1,0] is A[1,0] -> idx 3
    # A_TL[1,1] is A[1,1] -> idx 4
    
    # Indices for B_TL (2x2) in B (3x5):
    # B_TL[0,0] is B[0,0] -> idx 0
    # B_TL[0,1] is B[0,1] -> idx 1
    # B_TL[1,0] is B[1,0] -> idx 5
    # B_TL[1,1] is B[1,1] -> idx 6
    
    # Output indices for C_TL (2x2) in C (5x3):
    # C_TL[0,0] is C[0,0] -> idx 0
    # C_TL[0,1] is C[0,1] -> idx 1
    # C_TL[1,0] is C[1,0] -> idx 3
    # C_TL[1,1] is C[1,1] -> idx 4
    
    # Strassen Terms for A_TL * B_TL
    # M1 = A00 B00
    U[0][col] = 1; V[0][col] = 1; W[0][col] = 1; col += 1
    # M2 = (A00+A11) B11
    U[0][col] = 1; U[4][col] = 1; V[6][col] = 1; W[0][col] = 1; W[4][col] = 1; col += 1
    # M3 = A10 (B01-B11)
    U[3][col] = 1; V[1][col] = 1; V[6][col] = -1; W[1][col] = 1; W[4][col] = 1; col += 1
    # M4 = A11 (B00+B10)
    U[4][col] = 1; V[0][col] = 1; V[5][col] = 1; W[3][col] = 1; W[4][col] = 1; col += 1
    # M5 = (A00+A10) B10
    U[0][col] = 1; U[3][col] = 1; V[5][col] = 1; W[0][col] = -1; W[3][col] = -1; col += 1
    # M6 = (A01-A11) (B00+B10)
    U[1][col] = 1; U[4][col] = -1; V[0][col] = 1; V[5][col] = 1; W[4][col] = 1; col += 1
    # M7 = (A00-A11) (B11-B01)
    U[0][col] = 1; U[4][col] = -1; V[6][col] = 1; V[1][col] = -1; W[0][col] = 1; W[4][col] = -1; col += 1
    
    # Schoolbook Terms for the rest
    
    # A_TR (2x1) * B_BL (1x2)
    # A_TR: A[0,2] (idx 2), A[1,2] (idx 5)
    # B_BL: B[2,0] (idx 10), B[2,1] (idx 11)
    # C_TL: C[0,0] (0), C[0,1] (1), C[1,0] (3), C[1,1] (4)
    # Term (i=0, j=2, k=0) -> C[0,0]
    add_schoolbook(0, 2, 0)
    # Term (i=0, j=2, k=1) -> C[0,1]
    add_schoolbook(0, 2, 1)
    # Term (i=1, j=2, k=0) -> C[1,0]
    add_schoolbook(1, 2, 0)
    # Term (i=1, j=2, k=1) -> C[1,1]
    add_schoolbook(1, 2, 1)
    
    # A_TL (2x2) * B_TR (2x3)
    # A_TL: A[0,0] (0), A[0,1] (1), A[1,0] (3), A[1,1] (4)
    # B_TR: B[0,2] (2), B[0,3] (3), B[0,4] (4), B[1,2] (7), B[1,3] (8), B[1,4] (9)
    # C_TR: C[0,2] (2), C[0,3] (3), C[0,4] (4), C[1,2] (5), C[1,3] (6), C[1,4] (7)
    # i in 0..1, j in 0..1, k in 2..4
    for i in range(2):
        for j in range(2):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
                
    # A_TR (2x1) * B_BR (1x3)
    # A_TR: A[0,2] (2), A[1,2] (5)
    # B_BR: B[2,2] (12), B[2,3] (13), B[2,4] (14)
    # C_TR: C[0,2] (2), C[0,3] (3), C[0,4] (4), C[1,2] (5), C[1,3] (6), C[1,4] (7)
    for i in range(2):
        for k in range(2, 5):
            add_schoolbook(i, 2, k)
            
    # A_BL (1x2) * B_TL (2x2)
    # A_BL: A[2,0] (6), A[2,1] (7)
    # B_TL: B[0,0] (0), B[0,1] (1), B[1,0] (5), B[1,1] (6)
    # C_BL: C[2,0] (6), C[2,1] (7), C[3,0] (9), C[3,1] (10)
    for i in range(2, 3):
        for j in range(2):
            for k in range(2):
                add_schoolbook(i, j, k)
                
    # A_BR (1x1) * B_BL (1x2)
    # A_BR: A[2,2] (8)
    # B_BL: B[2,0] (10), B[2,1] (11)
    # C_BL: C[2,0] (6), C[2,1] (7)
    for i in range(2, 3):
        for k in range(2):
            add_schoolbook(i, 2, k)
            
    # A_BL (1x2) * B_TR (2x3)
    # A_BL: A[2,0] (6), A[2,1] (7)
    # B_TR: B[0,2] (2), B[0,3] (3), B[0,4] (4), B[1,2] (7), B[1,3] (8), B[1,4] (9)
    # C_BR: C[2,2] (8), C[2,3] (9), C[2,4] (10), C[3,2] (11), C[3,3] (12), C[3,4] (13)
    for i in range(2, 3):
        for j in range(2):
            for k in range(2, 5):
                add_schoolbook(i, j, k)
                
    # A_BR (1x1) * B_BR (1x3)
    # A_BR: A[2,2] (8)
    # B_BR: B[2,2] (12), B[2,3] (13), B[2,4] (14)
    # C_BR: C[2,2] (8), C[2,3] (9), C[2,4] (10)
    for i in range(2, 3):
        for k in range(2, 5):
            add_schoolbook(i, 2, k)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
