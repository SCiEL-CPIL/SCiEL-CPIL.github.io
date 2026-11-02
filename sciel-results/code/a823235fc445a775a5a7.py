"""Tensor decomposition for 2x3x8 matrix multiplication using a block decomposition with Strassen's algorithm."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 44: 28 from 2x2x8 (4 blocks of 2x2x2 Strassen) + 16 from 2x1x8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # a_idx: indices in U for A elements (4 elements)
    # b_idx: indices in V for B elements (4 elements)
    # c_idx: indices in W for C elements (4 elements)
    def add_strassen_block(a_idx, b_idx, c_idx):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        # C00 += M1, C11 += M1
        U[a_idx[0]][col] += 1
        U[a_idx[2]][col] += 1
        V[b_idx[0]][col] += 1
        V[b_idx[2]][col] += 1
        W[c_idx[0]][col] += 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        # C10 += M2
        U[a_idx[1]][col] += 1
        U[a_idx[3]][col] += 1
        V[b_idx[0]][col] += 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # M3 = A00(B01-B11)
        # C01 += M3
        U[a_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        V[b_idx[3]][col] -= 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M4 = A11(B10-B00)
        # C00 += M4, C10 += M4
        U[a_idx[3]][col] += 1
        V[b_idx[2]][col] += 1
        V[b_idx[0]][col] -= 1
        W[c_idx[0]][col] += 1
        W[c_idx[1]][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        # C00 -= M5, C01 += M5
        U[a_idx[0]][col] += 1
        U[a_idx[1]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] -= 1
        W[c_idx[2]][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        # C11 += M6
        U[a_idx[1]][col] += 1
        U[a_idx[0]][col] -= 1
        V[b_idx[0]][col] += 1
        V[b_idx[1]][col] += 1
        W[c_idx[3]][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        # C00 += M7, C11 -= M7 (Wait, C11 = M1 - M2 + M3 + M6? No)
        # Strassen:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # Let's re-verify M7 contribution.
        # M7 is (A01-A11)(B10+B11).
        # It contributes to C00 and C11?
        # Standard Strassen:
        # p1 = (a11+a22)(b11+b22)
        # p2 = (a21+a22)b11
        # p3 = a11(b12-b22)
        # p4 = a22(b21-b11)
        # p5 = (a11+a12)b22
        # p6 = (a21-a11)(b11+b12)
        # p7 = (a12-a22)(b21+b22)
        # c11 = p1 + p4 - p5 + p7
        # c12 = p3 + p5
        # c21 = p2 + p4
        # c22 = p1 - p2 + p3 + p6
        
        # So M7 (p7) contributes to C00 (+1) and C11 (0)?
        # Wait, c22 does NOT have p7.
        # c11 has p7.
        # So W[c_idx[0]][col] += 1.
        U[a_idx[1]][col] += 1
        U[a_idx[3]][col] -= 1
        V[b_idx[2]][col] += 1
        V[b_idx[3]][col] += 1
        W[c_idx[0]][col] += 1
        col += 1

    # A is 2x3. Indices:
    # A00: 0, A01: 1, A02: 2
    # A10: 3, A11: 4, A12: 5
    
    # B is 3x8. Indices:
    # Row 0: 0..7
    # Row 1: 8..15
    # Row 2: 16..23
    
    # C is 2x8. Indices:
    # Row 0: 0..7
    # Row 1: 8..15
    
    # Block 1: A_L (A00,A01,A10,A11) * B_L1 (B00,B01,B10,B11) -> C_L1 (C00,C01,C10,C11)
    # A_L indices: [0, 1, 3, 4] (A00, A01, A10, A11)
    # B_L1 indices: [0, 1, 8, 9] (B00, B01, B10, B11)
    # C_L1 indices: [0, 1, 8, 9] (C00, C01, C10, C11)
    add_strassen_block([0, 1, 3, 4], [0, 1, 8, 9], [0, 1, 8, 9])
    
    # Block 2: A_L * B_L2a (B02,B03,B12,B13) -> C_L2a (C02,C03,C12,C13)
    add_strassen_block([0, 1, 3, 4], [2, 3, 10, 11], [2, 3, 10, 11])
    
    # Block 3: A_L * B_L2b1 (B04,B05,B14,B15) -> C_L2b1 (C04,C05,C14,C15)
    add_strassen_block([0, 1, 3, 4], [4, 5, 12, 13], [4, 5, 12, 13])
    
    # Block 4: A_L * B_L2b2 (B06,B07,B16,B17) -> C_L2b2 (C06,C07,C16,C17)
    add_strassen_block([0, 1, 3, 4], [6, 7, 14, 15], [6, 7, 14, 15])
    
    # Term 2: A_R * B_R
    # A_R is col 2 of A: A02 (idx 2), A12 (idx 5)
    # B_R is row 2 of B: B20..B27 (idx 16..23)
    # C is 2x8. C0k (idx k), C1k (idx 8+k)
    
    for k in range(8):
        # A02 * B2k -> C0k
        U[2][col] = 1
        V[16 + k][col] = 1
        W[k][col] = 1
        col += 1
        
        # A12 * B2k -> C1k
        U[5][col] = 1
        V[16 + k][col] = 1
        W[8 + k][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
