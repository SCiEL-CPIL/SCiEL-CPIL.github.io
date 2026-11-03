import time
import numpy as np

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Shapes
    dim_A = n * m  # 6
    dim_B = m * p  # 12
    dim_C = p * n  # 8
    
    rank = 22
    
    U = np.zeros((dim_A, rank), dtype=float)
    V = np.zeros((dim_B, rank), dtype=float)
    W = np.zeros((dim_C, rank), dtype=float)
    
    # Block 1: A1 (idx 0,3) * B1_left (idx 0,1) -> C_left (idx 0,1,2,3)
    # Rank 4 (Schoolbook for 2x1x2)
    # r=0: A0 B0 -> C00 (idx 0)
    U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
    # r=1: A0 B1 -> C01 (idx 2)
    U[0, 1] = 1; V[1, 1] = 1; W[2, 1] = 1
    # r=2: A1 B0 -> C10 (idx 1)
    U[3, 2] = 1; V[0, 2] = 1; W[1, 2] = 1
    # r=3: A1 B1 -> C11 (idx 3)
    U[3, 3] = 1; V[1, 3] = 1; W[3, 3] = 1
    
    # Block 2: A1 (idx 0,3) * B1_right (idx 2,3) -> C_right (idx 4,5,6,7)
    # Rank 4
    # r=4: A0 B2 -> C02 (idx 4)
    U[0, 4] = 1; V[2, 4] = 1; W[4, 4] = 1
    # r=5: A0 B3 -> C03 (idx 6)
    U[0, 5] = 1; V[3, 5] = 1; W[6, 5] = 1
    # r=6: A1 B2 -> C12 (idx 5)
    U[3, 6] = 1; V[2, 6] = 1; W[5, 6] = 1
    # r=7: A1 B3 -> C13 (idx 7)
    U[3, 7] = 1; V[3, 7] = 1; W[7, 7] = 1
    
    # Block 3: A2 (idx 1,2,4,5) * B2_left (idx 4,5,6,7) -> C_mid (idx 0,1,2,3)
    # Rank 7 (Strassen)
    # A2: [[A1, A2], [A4, A5]] (indices 1,2,4,5)
    # B2: [[B4, B5], [B6, B7]] (indices 4,5,6,7)
    # C:  [[C0, C2], [C1, C3]] (indices 0,2,1,3)
    
    # M1 = A1 (B4 + B7)
    U[1, 8] = 1; V[4, 8] = 1; V[7, 8] = 1
    W[2, 8] = 1; W[3, 8] = 1  # C01, C11
    
    # M2 = (A1 + A4) B7
    U[1, 9] = 1; U[4, 9] = 1; V[7, 9] = 1
    W[2, 9] = 1; W[0, 9] = -1  # C01, C00
    
    # M3 = (A2 - A1) B4
    U[2, 10] = 1; U[1, 10] = -1; V[4, 10] = 1
    W[1, 10] = 1; W[3, 10] = -1  # C10, C11
    
    # M4 = A2 (B6 + B4)
    U[2, 11] = 1; V[6, 11] = 1; V[4, 11] = 1
    W[0, 11] = 1; W[1, 11] = 1  # C00, C10
    
    # M5 = (A1 + A2) B7
    U[1, 12] = 1; U[2, 12] = 1; V[7, 12] = 1
    W[0, 12] = 1; W[3, 12] = 1  # C00, C11
    
    # M6 = (A2 - A5) (B4 + B6)
    U[2, 13] = 1; U[5, 13] = -1; V[4, 13] = 1; V[6, 13] = 1
    W[0, 13] = 1  # C00
    
    # M7 = (A1 - A4) (B7 + B5)
    U[1, 14] = 1; U[4, 14] = -1; V[7, 14] = 1; V[5, 14] = 1
    W[3, 14] = -1  # C11
    
    # Block 4: A2 (idx 1,2,4,5) * B2_right (idx 8,9,10,11) -> C_right2 (idx 4,5,6,7)
    # Rank 7 (Strassen)
    # B2_right: [[B8, B9], [B10, B11]]
    # C: [[C4, C6], [C5, C7]] (indices 4,6,5,7)
    
    # M1' = A1 (B8 + B11)
    U[1, 15] = 1; V[8, 15] = 1; V[11, 15] = 1
    W[6, 15] = 1; W[7, 15] = 1  # C03, C13
    
    # M2' = (A1 + A4) B11
    U[1, 16] = 1; U[4, 16] = 1; V[11, 16] = 1
    W[6, 16] = 1; W[4, 16] = -1  # C03, C02
    
    # M3' = (A2 - A1) B8
    U[2, 17] = 1; U[1, 17] = -1; V[8, 17] = 1
    W[5, 17] = 1; W[7, 17] = -1  # C12, C13
    
    # M4' = A2 (B10 + B8)
    U[2, 18] = 1; V[10, 18] = 1; V[8, 18] = 1
    W[4, 18] = 1; W[5, 18] = 1  # C02, C12
    
    # M5' = (A1 + A2) B11
    U[1, 19] = 1; U[2, 19] = 1; V[11, 19] = 1
    W[4, 19] = 1; W[7, 19] = 1  # C02, C13
    
    # M6' = (A2 - A5) (B8 + B10)
    U[2, 20] = 1; U[5, 20] = -1; V[8, 20] = 1; V[10, 20] = 1
    W[4, 20] = 1  # C02
    
    # M7' = (A1 - A4) (B11 + B9)
    U[1, 21] = 1; U[4, 21] = -1; V[11, 21] = 1; V[9, 21] = 1
    W[7, 21] = -1  # C13
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
