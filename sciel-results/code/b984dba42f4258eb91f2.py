"""Rewritten program with Rank 50 decomposition using Strassen blocks."""
DIMENSIONS = [2, 4, 7]

def _strassen_2x2x2():
    """Returns U, V, W for 2x2x2 Strassen decomposition (Rank 7)."""
    # U: 4 rows (A elements), V: 4 rows (B elements), W: 4 rows (C elements)
    # Standard Strassen 7 terms
    U = [
        [1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1]
    ]
    V = [
        [1, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0] # Wait, Strassen V is for B.
    ]
    # Correct Strassen 7 factors for C = AB
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # U (A coeffs)
    U = [
        [1, 1, 0, 0, 1, 1, 0], # a00
        [0, 0, 1, 0, 0, 0, 1], # a01
        [0, 1, 0, 0, 0, 0, -1],# a10
        [1, 1, 0, 1, 0, 0, -1] # a11
    ]
    # V (B coeffs)
    V = [
        [1, 0, 1, -1, 0, 1, 1], # b00
        [0, 1, -1, 0, 0, 1, 0], # b01
        [1, 0, 0, 1, 1, 0, 1],  # b10
        [1, 1, -1, 0, 1, 0, 0]  # b11
    ]
    # W (C coeffs)
    # c00 = M1 - M4 + M6 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M3 + M5 - M7
    W = [
        [1, -1, 0, 0, 0, 0, 0], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, 0, -1, 0, 1, 0, -1] # c11
    ]
    return U, V, W

def _schoolbook_2x2x1():
    """Returns U, V, W for 2x2x1 Schoolbook decomposition (Rank 4)."""
    # A is 2x2, B is 2x1, C is 2x1
    # c0 = a00*b0 + a01*b1
    # c1 = a10*b0 + a11*b1
    U = [
        [1, 1, 0, 0], # a00
        [0, 0, 1, 0], # a01? No, a10
        [0, 0, 0, 1]  # a11
    ]
    # Wait, U has 4 rows for A (a00, a01, a10, a11)
    U = [
        [1, 1, 0, 0], # a00
        [0, 0, 0, 0], # a01 (not used in c0/c1 directly? No, c0 uses a01)
        [0, 0, 1, 1], # a10
        [0, 0, 0, 1]  # a11
    ]
    # Correct U for 2x2x1:
    # Term 1: a00 b0 -> c0
    # Term 2: a01 b1 -> c0
    # Term 3: a10 b0 -> c1
    # Term 4: a11 b1 -> c1
    U = [
        [1, 0, 0, 0], # a00
        [0, 1, 0, 0], # a01
        [0, 0, 1, 0], # a10
        [0, 0, 0, 1]  # a11
    ]
    V = [
        [1, 0, 1, 0], # b0
        [0, 1, 0, 1]  # b1
    ]
    W = [
        [1, 1, 0, 0], # c0
        [0, 0, 1, 1]  # c1
    ]
    return U, V, W

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct Rank 50.
    # Split p=7 into 2, 2, 2, 1.
    # For each segment of p, we split m=4 into 2, 2.
    # So we have 3 segments of 2 (using Strassen 2x2x2) and 1 segment of 1 (using Schoolbook 2x2x1).
    
    U_rows = n * m # 8
    V_rows = m * p # 28
    W_rows = p * n # 14
    
    U = [[0] * 50 for _ in range(U_rows)]
    V = [[0] * 50 for _ in range(V_rows)]
    W = [[0] * 50 for _ in range(W_rows)]
    
    col_offset = 0
    sU, sV, sW = _strassen_2x2x2()
    bU, bV, bW = _schoolbook_2x2x1()
    
    # Segment 1: p[0:2]
    # A is 2x4, B is 4x2.
    # Split A into A_L (rows 0-1, cols 0-1) and A_R (rows 0-1, cols 2-3).
    # Split B into B_T (rows 0-1) and B_B (rows 2-3).
    # Product C1 = A_L B_T + A_R B_B.
    
    # Term 1: A_L B_T (2x2 x 2x2 -> 2x2)
    for r in range(4): # U local rows (A_L)
        for c in range(7): # Strassen rank
            U[r][col_offset + c] = sU[r][c]
    for r in range(4): # V local rows (B_T)
        for c in range(7):
            V[r][col_offset + c] = sV[r][c]
    for r in range(4): # W local rows (C1[0:2])
        for c in range(7):
            W[r][col_offset + c] = sW[r][c]
    col_offset += 7
    
    # Term 2: A_R B_B (2x2 x 2x2 -> 2x2)
    for r in range(4): # U local rows (A_R, global rows 2-3)
        for c in range(7):
            U[r + 2][col_offset + c] = sU[r][c]
    for r in range(4): # V local rows (B_B, global rows 2-3)
        for c in range(7):
            V[r + 2][col_offset + c] = sV[r][c]
    for r in range(4): # W local rows (C1[0:2])
        for c in range(7):
            W[r][col_offset + c] = sW[r][c]
    col_offset += 7
    
    # Segment 2: p[2:4]
    # Same logic, but W writes to C[2:4] (global W indices 2,3,4,5)
    for r in range(4):
        for c in range(7):
            U[r][col_offset + c] = sU[r][c]
    for r in range(4):
        for c in range(7):
            V[r + 4][col_offset + c] = sV[r][c]
    for r in range(4):
        for c in range(7):
            W[r + 4][col_offset + c] = sW[r][c]
    col_offset += 7
    
    for r in range(4):
        for c in range(7):
            U[r + 2][col_offset + c] = sU[r][c]
    for r in range(4):
        for c in range(7):
            V[r + 6][col_offset + c] = sV[r][c]
    for r in range(4):
        for c in range(7):
            W[r + 4][col_offset + c] = sW[r][c]
    col_offset += 7
    
    # Segment 3: p[4:6]
    for r in range(4):
        for c in range(7):
            U[r][col_offset + c] = sU[r][c]
    for r in range(4):
        for c in range(7):
            V[r + 8][col_offset + c] = sV[r][c]
    for r in range(4):
        for c in range(7):
            W[r + 8][col_offset + c] = sW[r][c]
    col_offset += 7
    
    for r in range(4):
        for c in range(7):
            U[r + 2][col_offset + c] = sU[r][c]
    for r in range(4):
        for c in range(7):
            V[r + 10][col_offset + c] = sV[r][c]
    for r in range(4):
        for c in range(7):
            W[r + 8][col_offset + c] = sW[r][c]
    col_offset += 7
    
    # Segment 4: p[6:7]
    # A is 2x4, B is 4x1.
    # C4 = A_L B_T + A_R B_B.
    # B_T is 2x1, B_B is 2x1.
    
    # Term 1: A_L B_T (2x2 x 2x1 -> 2x1)
    for r in range(4):
        for c in range(4):
            U[r][col_offset + c] = bU[r][c]
    for r in range(2): # V local rows (B_T)
        for c in range(4):
            V[r + 12][col_offset + c] = bV[r][c]
    for r in range(2): # W local rows (C4[0:2])
        for c in range(4):
            W[r + 12][col_offset + c] = bW[r][c]
    col_offset += 4
    
    # Term 2: A_R B_B (2x2 x 2x1 -> 2x1)
    for r in range(4):
        for c in range(4):
            U[r + 2][col_offset + c] = bU[r][c]
    for r in range(2):
        for c in range(4):
            V[r + 14][col_offset + c] = bV[r][c]
    for r in range(2):
        for c in range(4):
            W[r + 12][col_offset + c] = bW[r][c]
    col_offset += 4
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

