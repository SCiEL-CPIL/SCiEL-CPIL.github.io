import time

DIMENSIONS = [4, 4, 8]


def _strassen_2x2():
    """Return (U, V, W) for 2x2x2 matrix multiplication, rank 7.
    U, V, W are lists of lists of size 4 x 7.
    U[i*2+j, c] is coefficient for A[i,j] in the c-th rank-1 factor.
    V[i*2+j, c] is coefficient for B[i,j] in the c-th rank-1 factor.
    W[i*2+j, c] is coefficient for C[i,j] in the c-th rank-1 factor.
    """
    n = 2
    rank = 7
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]

    # 7 Strassen products
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Factor 0: M1
    U[0][0] = 1  # A00
    U[3][0] = 1  # A11
    V[0][0] = 1  # B00
    V[3][0] = 1  # B11
    W[0][0] = 1  # C00
    W[3][0] = 1  # C11

    # Factor 1: M2
    U[2][1] = 1  # A10
    U[3][1] = 1  # A11
    V[0][1] = 1  # B00
    W[2][1] = 1  # C10
    W[3][1] = -1  # C11

    # Factor 2: M3
    U[0][2] = 1  # A00
    V[1][2] = 1  # B01
    V[3][2] = -1  # B11
    W[1][2] = 1  # C01
    W[3][2] = 1  # C11

    # Factor 3: M4
    U[3][3] = 1  # A11
    V[2][3] = 1  # B10
    V[0][3] = -1  # B00
    W[0][3] = 1  # C00
    W[2][3] = 1  # C10

    # Factor 4: M5
    U[0][4] = 1  # A00
    U[1][4] = 1  # A01
    V[3][4] = 1  # B11
    W[0][4] = -1  # C00
    W[1][4] = 1  # C01

    # Factor 5: M6
    U[2][5] = 1  # A10
    U[0][5] = -1  # A00
    V[0][5] = 1  # B00
    V[1][5] = 1  # B01
    W[3][5] = 1  # C11

    # Factor 6: M7
    U[1][6] = 1  # A01
    U[3][6] = -1  # A11
    V[2][6] = 1  # B10
    V[3][6] = 1  # B11
    W[0][6] = 1  # C00

    return U, V, W


def _block_4x4():
    """Return (U, V, W) for 4x4x4 matrix multiplication, rank 49.
    Uses two Strassen 2x2 products on 2x2 blocks of size 2.
    U, V, W are lists of lists of size 16 x 49.
    """
    n = 4
    rank = 49
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]

    S2_U, S2_V, S2_W = _strassen_2x2()

    # A partitioned into 4 blocks: A00, A01, A10, A11 (each 2x2)
    # B partitioned into 4 blocks: B00, B01, B10, B11 (each 2x2)
    # C partitioned into 4 blocks: C00, C01, C10, C11 (each 2x2)

    # We perform two independent Strassen multiplications:
    # First: compute (A00+A11)(B00+B11) + (A11)(B10-B00) + (A00)(B01-B11) + (A01-A11)(B10+B11) ... 
    # Actually, the standard approach: treat the 4x4 multiplication as two 2x2 multiplications
    # on the block structure, but that's not quite right either.

    # Actually, for 4x4x4, we can view it as 2x2 blocks of 2x2 matrices.
    # The multiplication is: C = A * B where A, B, C are 2x2 block matrices of 2x2 entries.
    # We can use Strassen on the block level. The 7 Strassen products are 2x2 matrix multiplications.
    # Each 2x2 multiplication has rank 7 (by Strassen).
    # So total rank = 7 * 7 = 49.

    # Let's build this properly.
    # Block indices: (bi, bj) where bi, bj in {0,1}
    # A_block[bi, bj] is a 2x2 matrix at position (2*bi : 2*bi+2, 2*bj : 2*bj+2)
    # Similarly for B and C.

    # Strassen at block level:
    # M1 = (A00+A11) * (B00+B11)
    # M2 = (A10+A11) * B00
    # M3 = A00 * (B01-B11)
    # M4 = A11 * (B10-B00)
    # M5 = (A00+A01) * B11
    # M6 = (A10-A00) * (B00+B01)
    # M7 = (A01-A11) * (B10+B11)

    # Each M_i is a 2x2 matrix product, which we compute with Strassen (rank 7).
    # The final C blocks are linear combinations of the M_i blocks.

    # For each Strassen product i (0..6), we need to compute a 2x2 matrix multiplication.
    # The "A" for product i is a linear combination of A blocks.
    # The "B" for product i is a linear combination of B blocks.
    # The result contributes to C blocks with certain coefficients.

    # Let's define the Strassen block-level products:
    # Product i: A_prod[i] = sum over (bi,bj) of coefA[i][bi][bj] * A_block[bi,bj]
    #            B_prod[i] = sum over (bi,bj) of coefB[i][bi][bj] * B_block[bi,bj]
    #            C_prod[i] = A_prod[i] * B_prod[i]  (2x2 matrix)
    #            C_block[ci,cj] += coefC[i][ci][cj] * C_prod[i]

    # Strassen coefficients (block level):
    # i=0 (M1): A: A00+A11, B: B00+B11, C: C00 += M1, C11 += M1
    # i=1 (M2): A: A10+A11, B: B00, C: C10 += M2, C11 -= M2
    # i=2 (M3): A: A00, B: B01-B11, C: C01 += M3, C11 += M3
    # i=3 (M4): A: A11, B: B10-B00, C: C00 += M4, C10 += M4
    # i=4 (M5): A: A00+A01, B: B11, C: C00 -= M5, C01 += M5
    # i=5 (M6): A: A10-A00, B: B00+B01, C: C11 += M6
    # i=6 (M7): A: A01-A11, B: B10+B11, C: C00 += M7

    # Now for each product i, the 2x2 matrix multiplication is done via Strassen.
    # The 2x2 Strassen has 7 rank-1 factors.
    # So for product i, factor t (0..6), we have:
    #   U contribution: for each (bi,bj) with coefA[i][bi][bj] != 0, and for each (a1,a2) in 2x2,
    #     the Strassen 2x2 U matrix element at (a1,a2) for factor t, multiplied by coefA[i][bi][bj]
    #   Similarly for V and W.

    # Let's map out the indices.
    # A matrix index: row = 2*bi + a1, col = 2*bj + a2, where bi,bj in {0,1}, a1,a2 in {0,1}
    # B matrix index: row = 2*bi + b1, col = 2*bj + b2
    # C matrix index: row = 2*ci + c1, col = 2*cj + c2

    # Strassen 2x2 U, V, W are 4x7 matrices.
    # U[a1*2+a2, t], V[b1*2+b2, t], W[c1*2+c2, t]

    # For product i, factor t:
    # U_full[2*bi + a1, 2*bj + a2, i*7 + t] += coefA[i][bi][bj] * S2_U[a1*2+a2, t]
    # V_full[2*bi + b1, 2*bj + b2, i*7 + t] += coefB[i][bi][bj] * S2_V[b1*2+b2, t]
    # W_full[2*ci + c1, 2*cj + c2, i*7 + t] += coefC[i][ci][cj] * S2_W[c1*2+c2, t]

    # Let's define the coefficients.
    # coefA[i][bi][bj]: coefficient of A_block[bi,bj] in A_prod[i]
    # coefB[i][bi][bj]: coefficient of B_block[bi,bj] in B_prod[i]
    # coefC[i][ci][cj]: coefficient of C_prod[i] in C_block[ci,cj]

    coefA = [
        {(0,0): 1, (1,1): 1},       # M1: A00 + A11
        {(1,0): 1, (1,1): 1},       # M2: A10 + A11
        {(0,0): 1},                  # M3: A00
        {(1,1): 1},                  # M4: A11
        {(0,0): 1, (0,1): 1},       # M5: A00 + A01
        {(1,0): 1, (0,0): -1},      # M6: A10 - A00
        {(0,1): 1, (1,1): -1},      # M7: A01 - A11
    ]

    coefB = [
        {(0,0): 1, (1,1): 1},       # M1: B00 + B11
        {(0,0): 1},                  # M2: B00
        {(0,1): 1, (1,1): -1},      # M3: B01 - B11
        {(1,0): 1, (0,0): -1},      # M4: B10 - B00
        {(1,1): 1},                  # M5: B11
        {(0,0): 1, (0,1): 1},       # M6: B00 + B01
        {(1,0): 1, (1,1): 1},       # M7: B10 + B11
    ]

    coefC = [
        {(0,0): 1, (1,1): 1},       # M1: C00 += M1, C11 += M1
        {(1,0): 1, (1,1): -1},      # M2: C10 += M2, C11 -= M2
        {(0,1): 1, (1,1): 1},       # M3: C01 += M3, C11 += M3
        {(0,0): 1, (1,0): 1},       # M4: C00 += M4, C10 += M4
        {(0,0): -1, (0,1): 1},      # M5: C00 -= M5, C01 += M5
        {(1,1): 1},                  # M6: C11 += M6
        {(0,0): 1},                  # M7: C00 += M7
    ]

    for i in range(7):
        for t in range(7):
            col = i * 7 + t
            # U
            for (bi, bj), ca in coefA[i].items():
                if ca == 0:
                    continue
                for a1 in range(2):
                    for a2 in range(2):
                        val = ca * S2_U[a1 * 2 + a2][t]
                        if val != 0:
                            row = 2 * bi + a1
                            col_a = 2 * bj + a2
                            U[row][col] += val
            # V
            for (bi, bj), cb in coefB[i].items():
                if cb == 0:
                    continue
                for b1 in range(2):
                    for b2 in range(2):
                        val = cb * S2_V[b1 * 2 + b2][t]
                        if val != 0:
                            row = 2 * bi + b1
                            col_b = 2 * bj + b2
                            V[row][col] += val
            # W
            for (ci, cj), cc in coefC[i].items():
                if cc == 0:
                    continue
                for c1 in range(2):
                    for c2 in range(2):
                        val = cc * S2_W[c1 * 2 + c2][t]
                        if val != 0:
                            row = 2 * ci + c1
                            col_c = 2 * cj + c2
                            W[row][col] += val

    return U, V, W


def construct():
    """Return {'U', 'V', 'W'} for 4x4x8 matrix multiplication.
    U: 16 x rank, V: 32 x rank, W: 32 x rank.
    Uses two copies of the 4x4x4 Strassen decomposition.
    Rank = 98.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    U4, V4, W4 = _block_4x4()
    
    rank = 98
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(32)]
    W = [[0] * rank for _ in range(32)]
    
    # First 49 columns: multiply A (4x4) by B1 (4x4, first half of B)
    # A is 4x4, B1 is 4x4 (columns 0-3 of B), C1 is 4x4 (columns 0-3 of C)
    # U[:, 0:49] = U4
    # V[0:16, 0:49] = V4
    # V[16:32, 0:49] = 0
    # W[0:16, 0:49] = W4
    # W[16:32, 0:49] = 0
    
    for i in range(16):
        for c in range(49):
            U[i][c] = U4[i][c]
            W[i][c] = W4[i][c]
            V[i][c] = V4[i][c]
    
    # Next 49 columns: multiply A (4x4) by B2 (4x4, second half of B)
    # A is 4x4, B2 is 4x4 (columns 4-7 of B), C2 is 4x4 (columns 4-7 of C)
    # U[:, 49:98] = U4
    # V[0:16, 49:98] = 0
    # V[16:32, 49:98] = V4
    # W[0:16, 49:98] = 0
    # W[16:32, 49:98] = W4
    
    for i in range(16):
        for c in range(49):
            col = 49 + c
            U[i][col] = U4[i][c]
            W[i + 16][col] = W4[i][c]
            V[i + 16][col] = V4[i][c]
    
    return {"U": U, "V": V, "W": W}
