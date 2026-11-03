"""
Matrix multiplication tensor decomposition for 3x4 * 4x6.
Uses Strassen (rank 7) for 2x2*2x2 blocks and direct (rank 4) for 1x2*2x2 blocks.
Total rank = 3*2*4 + 3*2*7 = 24 + 42 = 66.
"""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    """
    Construct factor matrices U, V, W for the 3x4x6 multiplication tensor.
    Returns dict with keys 'U', 'V', 'W'.
    """
    start_time = time.time()
    n, m, p = 3, 4, 6
    
    # Target rank:
    # C is 3x6.
    # Split C into 3 column pairs (each 3x2).
    # For each 3x2 block of C:
    #   C_block = A * B_block
    #   A is 3x4, B_block is 4x2.
    #   Split A into rows 0 (1x4) and rows 1-2 (2x4).
    #   Split B_block into rows 0-1 (2x2) and rows 2-3 (2x2).
    #   C_block = A_top * B_top + A_bot * B_bot
    #   A_top is 1x4, B_top is 2x2 -> 1x2*2x2 multiplication (rank 4).
    #   A_bot is 2x4, B_bot is 2x2 -> 2x2*2x2 multiplication (rank 7).
    #   Total per column pair: 4 + 7 = 11.
    #   Total for 3 column pairs: 3 * 11 = 33? 
    #   Wait, the previous inspiration said 66. Let's re-read.
    #   Inspiration 1: "Row 0 of C: 3 pairs of (1x2 * 2x2) blocks. Each 1x2*2x2 is rank 4. Total 3*2*4 = 24."
    #   "Rows 1-2 of C: 3 pairs of (2x2 * 2x2) blocks. Each 2x2*2x2 is rank 7 (Strassen). Total 3*2*7 = 42."
    #   Total rank = 24 + 42 = 66.
    
    # Why 3*2*4?
    # C[0, :] has 3 column pairs. For each pair, we compute C[0, 2j:2j+2].
    # This requires A[0, :] * B[:, 2j:2j+2].
    # A[0, :] is 1x4. B[:, 2j:2j+2] is 4x2.
    # Split B into B_top (2x2) and B_bot (2x2).
    # C_part = A[0,0:2]*B_top + A[0,2:4]*B_bot.
    # Each term is 1x2 * 2x2, rank 4.
    # So for one column pair, we have two 1x2*2x2 terms. Total rank 8 for row 0.
    # For 3 column pairs, total rank 24.
    
    # Similarly for rows 1-2 of C.
    # C[1:3, 2j:2j+2] = A[1:3, :] * B[:, 2j:2j+2].
    # A[1:3, :] is 2x4. B is 4x2.
    # C_part = A[1:3,0:2]*B_top + A[1:3,2:4]*B_bot.
    # Each term is 2x2 * 2x2, rank 7 (Strassen).
    # So for one column pair, we have two 2x2*2x2 terms. Total rank 14 for rows 1-2.
    # For 3 column pairs, total rank 42.
    
    # Total rank = 24 + 42 = 66.
    
    R = 66
    U = [[0] * R for _ in range(n * m)]  # (12, 66)
    V = [[0] * R for _ in range(m * p)]  # (24, 66)
    W = [[0] * R for _ in range(p * n)]  # (18, 66)
    
    col = 0
    
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add Strassen 2x2*2x2 block at given offsets."""
        nonlocal col
        # A indices
        A00 = a_r * m + a_c
        A01 = a_r * m + a_c + 1
        A10 = (a_r + 1) * m + a_c
        A11 = (a_r + 1) * m + a_c + 1
        
        # B indices
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        # C indices (Trace convention: W[k*n + i, r])
        # c_r is row i, c_c is col k
        C00 = c_c * n + c_r
        C01 = (c_c + 1) * n + c_r
        C10 = c_c * n + (c_r + 1)
        C11 = (c_c + 1) * n + (c_r + 1)
        
        # M1 = (A00+A11)(B00+B11)
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1; W[C11][col] -= 1
        col += 1
        # M3 = A00(B01-B11)
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1; W[C11][col] += 1
        col += 1
        # M4 = A11(B10-B00)
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1
        col += 1
    
    def add_1x2_2x2(a_r, a_c, b_r, b_c, c_r, c_c):
        """Add 1x2*2x2 block (rank 4) at given offsets."""
        nonlocal col
        # A indices (1x2)
        A0 = a_r * m + a_c
        A1 = a_r * m + a_c + 1
        
        # B indices (2x2)
        B00 = b_r * p + b_c
        B01 = b_r * p + b_c + 1
        B10 = (b_r + 1) * p + b_c
        B11 = (b_r + 1) * p + b_c + 1
        
        # C indices (1x2)
        C0 = c_c * n + c_r
        C1 = (c_c + 1) * n + c_r
        
        # M1: A0*B00
        U[A0][col] += 1; V[B00][col] += 1; W[C0][col] += 1
        col += 1
        # M2: A1*B10
        U[A1][col] += 1; V[B10][col] += 1; W[C0][col] += 1
        col += 1
        # M3: A0*B01
        U[A0][col] += 1; V[B01][col] += 1; W[C1][col] += 1
        col += 1
        # M4: A1*B11
        U[A1][col] += 1; V[B11][col] += 1; W[C1][col] += 1
        col += 1
    
    # Build decomposition:
    # For each of the 3 column pairs (c_c in 0, 2, 4):
    
    # Row 0 of C:
    # C[0, c_c:c_c+2] = A[0, 0:2]*B[0:2, c_c:c_c+2] + A[0, 2:4]*B[2:4, c_c:c_c+2]
    for c_c in [0, 2, 4]:
        add_1x2_2x2(0, 0, 0, c_c, 0, c_c)   # 1x2*2x2, rank 4
        add_1x2_2x2(0, 2, 2, c_c, 0, c_c)   # 1x2*2x2, rank 4
        
    # Rows 1-2 of C:
    # C[1:3, c_c:c_c+2] = A[1:3, 0:2]*B[0:2, c_c:c_c+2] + A[1:3, 2:4]*B[2:4, c_c:c_c+2]
    for c_c in [0, 2, 4]:
        add_strassen(1, 0, 0, c_c, 1, c_c)   # 2x2*2x2, rank 7
        add_strassen(1, 2, 2, c_c, 1, c_c)   # 2x2*2x2, rank 7
    
    assert col == 66, f"Expected 66 columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
