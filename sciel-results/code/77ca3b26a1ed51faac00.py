import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    
    # Ranks:
    # 3 blocks of (1x2 * 2x2) for row 0 of C. Rank of 1x2*2x2 is 4 (schoolbook).
    # 6 blocks of (2x2 * 2x2) for rows 1,2 of C. Rank of 2x2*2x2 is 7 (Strassen).
    # Total rank = 3 * 4 + 6 * 7 = 12 + 42 = 54? 
    # Wait, the previous calculation in the prompt was 24 + 42 = 66.
    # Let's re-verify the partition in the prompt's seed logic.
    # Prompt says:
    # A (3x4) -> rows 1+2, cols 2+2
    # B (4x6) -> rows 2+2, cols 2+2+2
    # C (3x6)
    # Row 0 of C: 3 pairs of (1x2 * 2x2) blocks. 
    #   Each "pair" corresponds to the two column blocks of A (cols 0-1 and 2-3).
    #   So for C[0, 0:2], we have A[0, 0:2]*B[0:2, 0:2] + A[0, 2:4]*B[2:4, 0:2].
    #   This is two separate 1x2*2x2 multiplications. Each is rank 4.
    #   So for each of the 3 column pairs in C, we have 2 * rank(1x2*2x2) = 8.
    #   Total for Row 0: 3 * 8 = 24.
    # Rows 1-2 of C: 3 pairs of (2x2 * 2x2) blocks.
    #   For each of the 3 column pairs in C, we have 2 separate 2x2*2x2 multiplications.
    #   Each is rank 7 (Strassen).
    #   So for each of the 3 column pairs, we have 2 * 7 = 14.
    #   Total for Rows 1-2: 3 * 14 = 42.
    # Total Rank = 24 + 42 = 66.
    
    R = 66
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen 2x2x2 decomposition
    # Indices:
    # A: (a_r, a_c) to (a_r+1, a_c+1)
    # B: (b_r, b_c) to (b_r+1, b_c+1)
    # C: (c_r, c_c) to (c_r+1, c_c+1)
    # Trace convention: W[k*n + i, r] where k is col, i is row.
    # So C[0,0] -> W[0*n + 0], C[0,1] -> W[1*n + 0], C[1,0] -> W[0*n + 1], C[1,1] -> W[1*n + 1]
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        
        # U indices
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        
        # V indices
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        # W indices
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00 -> C10 += 1, C11 -= 1
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        
        # M3 = A00(B01-B11) -> C01 += 1, C11 += 1
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        
        # M4 = A11(B10-B00) -> C00 += 1, C10 += 1
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11 -> C00 -= 1, C01 += 1
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01) -> C11 += 1
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11) -> C00 += 1
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1

    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        nonlocal col
        
        # U indices (1x2)
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        
        # V indices (2x2)
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        # W indices (1x2)
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        # M1: A0*B00 -> C0 += 1
        U[A0][col] += 1
        V[B00][col] += 1
        W[C0][col] += 1
        col += 1
        
        # M2: A1*B10 -> C0 += 1
        U[A1][col] += 1
        V[B10][col] += 1
        W[C0][col] += 1
        col += 1
        
        # M3: A0*B01 -> C1 += 1
        U[A0][col] += 1
        V[B01][col] += 1
        W[C1][col] += 1
        col += 1
        
        # M4: A1*B11 -> C1 += 1
        U[A1][col] += 1
        V[B11][col] += 1
        W[C1][col] += 1
        col += 1

    # Construction Order:
    
    # Row 0 of C (1x6)
    # C[0, 0:2] = A[0, 0:2]*B[0:2, 0:2] + A[0, 2:4]*B[2:4, 0:2]
    add_1x2_2x2(0, 0, 0, 0, 0, 0)
    add_1x2_2x2(0, 2, 2, 0, 0, 0)
    
    # C[0, 2:4] = A[0, 0:2]*B[0:2, 2:4] + A[0, 2:4]*B[2:4, 2:4]
    add_1x2_2x2(0, 0, 0, 2, 0, 2)
    add_1x2_2x2(0, 2, 2, 2, 0, 2)
    
    # C[0, 4:6] = A[0, 0:2]*B[0:2, 4:6] + A[0, 2:4]*B[2:4, 4:6]
    add_1x2_2x2(0, 0, 0, 4, 0, 4)
    add_1x2_2x2(0, 2, 2, 4, 0, 4)
    
    # Rows 1-2 of C (2x6)
    # C[1:3, 0:2] = A[1:3, 0:2]*B[0:2, 0:2] + A[1:3, 2:4]*B[2:4, 0:2]
    add_strassen(1, 0, 0, 0, 1, 0)
    add_strassen(1, 2, 2, 0, 1, 0)
    
    # C[1:3, 2:4] = A[1:3, 0:2]*B[0:2, 2:4] + A[1:3, 2:4]*B[2:4, 2:4]
    add_strassen(1, 0, 0, 2, 1, 2)
    add_strassen(1, 2, 2, 2, 1, 2)
    
    # C[1:3, 4:6] = A[1:3, 0:2]*B[0:2, 4:6] + A[1:3, 2:4]*B[2:4, 4:6]
    add_strassen(1, 0, 0, 4, 1, 4)
    add_strassen(1, 2, 2, 4, 1, 4)
    
    if col != R:
        raise ValueError(f"Rank mismatch: expected {R}, got {col}")
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
