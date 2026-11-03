"""Optimized tensor decomposition for 2x3 x 3x8 matrix multiplication."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-44 exact tensor decomposition for 2x3 by 3x8 matrix multiplication.
    
    Strategy:
    - Split the middle dimension (size 3) into a left block of size 2 and a right part of size 1.
    - For A_L (2x2) and B_L (2x8): apply Strassen to 2x2x2 blocks, 4 blocks x 7 terms = 28 terms.
    - For A_R (2x1) and B_R (1x8): schoolbook, 2x1x8 = 16 terms.
    - Total rank = 28 + 16 = 44.
    """
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]   # 6 x 44
    V = [[0] * rank for _ in range(m * p)]   # 24 x 44
    W = [[0] * rank for _ in range(p * n)]   # 16 x 44

    col = 0

    # Strassen 2x2x2 terms: for the left block (columns 0,1 of A, columns 0..7 of B in pairs)
    # Strassen formulas for C = A*B where A is 2x2, B is 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A11-A10)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M2+M3-M4+M6
    # C10 = M2+M3+M5-M7
    # C11 = M1-M2+M4+M6

    # Strassen contribution to C from each term:
    # term 0 (M1): C00 += 1, C11 += -1
    # term 1 (M2): C01 += 1, C10 += 1
    # term 2 (M3): C01 += 1, C10 += 1
    # term 3 (M4): C00 += 1, C01 += -1, C11 += 1
    # term 4 (M5): C00 += -1, C10 += 1
    # term 5 (M6): C01 += 1, C11 += 1
    # term 6 (M7): C00 += 1, C10 += -1

    # For each 2x2 block b (0,1,2,3), B columns are (2b, 2b+1)
    for b in range(4):
        # Local A indices: (0,0)->(i=0,j=0), (0,1)->(i=0,j=1), (1,0)->(i=1,j=0), (1,1)->(i=1,j=1)
        # Global A index: i*m + j where m=3, so j is 0 or 1 (left block)
        a_00 = 0 * 3 + 0   # 0
        a_01 = 0 * 3 + 1   # 1
        a_10 = 1 * 3 + 0   # 3
        a_11 = 1 * 3 + 1   # 4

        # Local B indices: (0,0)->(j=0,k=2b), (0,1)->(j=0,k=2b+1), (1,0)->(j=1,k=2b), (1,1)->(j=1,k=2b+1)
        # Global B index: j*p + k where p=8
        b_00 = 0 * 8 + 2 * b
        b_01 = 0 * 8 + 2 * b + 1
        b_10 = 1 * 8 + 2 * b
        b_11 = 1 * 8 + 2 * b + 1

        # Local C indices: (0,0)->(i=0,k=2b), (0,1)->(i=0,k=2b+1), (1,0)->(i=1,k=2b), (1,1)->(i=1,k=2b+1)
        # Global C index: k*n + i where n=2
        c_00 = 2 * b * 2 + 0
        c_01 = 2 * b + 1 * 2 + 0
        c_10 = 2 * b * 2 + 1
        c_11 = 2 * b + 1 * 2 + 1

        # Term 0: M1 = (A00+A11)(B00+B11)
        U[a_00][col] = 1
        U[a_11][col] = 1
        V[b_00][col] = 1
        V[b_11][col] = 1
        W[c_00][col] = 1
        W[c_11][col] = -1
        col += 1

        # Term 1: M2 = (A10+A11)B00
        U[a_10][col] = 1
        U[a_11][col] = 1
        V[b_00][col] = 1
        W[c_01][col] = 1
        W[c_10][col] = 1
        col += 1

        # Term 2: M3 = A00(B10-B11)
        U[a_00][col] = 1
        V[b_10][col] = 1
        V[b_11][col] = -1
        W[c_01][col] = 1
        W[c_10][col] = 1
        col += 1

        # Term 3: M4 = A11(B01-B10)
        U[a_11][col] = 1
        V[b_01][col] = 1
        V[b_10][col] = -1
        W[c_00][col] = 1
        W[c_01][col] = -1
        W[c_11][col] = 1
        col += 1

        # Term 4: M5 = (A00+A10)B11
        U[a_00][col] = 1
        U[a_10][col] = 1
        V[b_11][col] = 1
        W[c_00][col] = -1
        W[c_10][col] = 1
        col += 1

        # Term 5: M6 = (A11-A10)(B00+B10)
        U[a_11][col] = 1
        U[a_10][col] = -1
        V[b_00][col] = 1
        V[b_10][col] = 1
        W[c_01][col] = 1
        W[c_11][col] = 1
        col += 1

        # Term 6: M7 = (A01-A11)(B01+B11)
        U[a_01][col] = 1
        U[a_11][col] = -1
        V[b_01][col] = 1
        V[b_11][col] = 1
        W[c_00][col] = 1
        W[c_10][col] = -1
        col += 1

    # Right part: A_R is column j=2 of A (size 2x1), B_R is row j=2 of B (size 1x8)
    # Schoolbook: for i in 0..1, k in 0..7: C[i,k] += A[i,2]*B[2,k]
    a_r = [0 * 3 + 2, 1 * 3 + 2]  # A[0,2]=2, A[1,2]=5
    b_r = [2 * 8 + k for k in range(8)]  # B[2,0..7] = 16..23

    for i in range(2):
        for k in range(8):
            U[a_r[i]][col] = 1
            V[b_r[k]][col] = 1
            W[k * 2 + i][col] = 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
