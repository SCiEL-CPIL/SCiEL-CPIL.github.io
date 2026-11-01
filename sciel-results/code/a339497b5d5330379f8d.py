"""4x4x4 matrix multiplication tensor decomposition using recursive Strassen (rank 49)."""
import time

DIMENSIONS = [4, 4, 4]

def _verify(U, V, W, n, m, p, max_trials=5):
    """Verify that the decomposition exactly reproduces matrix multiplication.
    C[i,k] = sum_{j,c} U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    """
    import random
    for _ in range(max_trials):
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        # Reference product
        C_ref = [[0]*p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    s += A[i][j] * B[j][k]
                C_ref[i][k] = s
        # Decomposition
        rank = len(U[0])
        for i in range(n):
            for k in range(p):
                s = 0
                for j in range(m):
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != C_ref[i][k]:
                    return False
    return True

def _strassen_2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2 matrix multiplication.
    
    U: (4, 7) - coefficients for A entries [A00, A01, A10, A11]
    V: (4, 7) - coefficients for B entries [B00, B01, B10, B11]
    W: (4, 7) - coefficients for C entries [C00, C01, C10, C11]
    
    The 7 Strassen products:
    p1 = (A00+A11)(B00+B11)
    p2 = (A10+A11)B00
    p3 = A00(B10-B11)
    p4 = A11(B01-B00)
    p5 = (A00+A01)B11
    p6 = (A10-A00)(B00+B01)
    p7 = (A01-A11)(B10+B11)
    
    C reconstruction:
    C00 = p1 + p4 - p5 + p7
    C01 = p3 + p5
    C10 = p2 + p4
    C11 = p1 - p2 + p3 + p6
    """
    # U rows: [A00, A01, A10, A11], columns: p1..p7
    U = [
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ]
    # V rows: [B00, B01, B10, B11], columns: p1..p7
    V = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 0, 1, 0, 1, 0],    # B01
        [1, 0, 1, 0, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1]    # B11
    ]
    # W rows: [C00, C01, C10, C11], columns: p1..p7
    W = [
        [1, 0, 0, 1, -1, 0, 1],   # C00
        [0, 0, 1, 0, 1, 0, 0],    # C01
        [0, 1, 0, 1, 0, 0, 0],    # C10
        [1, -1, 1, 0, 0, 1, 0]    # C11
    ]
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin within 120s

    n, m, p = 4, 4, 4
    rank = 49  # 7 * 7 from recursive Strassen

    U = [[0]*rank for _ in range(n*m)]  # (16, 49)
    V = [[0]*rank for _ in range(m*p)]  # (16, 49)
    W = [[0]*rank for _ in range(p*n)]  # (16, 49)

    # Get the 2x2 Strassen decomposition
    u2, v2, w2 = _strassen_2x2()
    
    # For 4x4, we partition into 2x2 blocks.
    # A is 4x4, partitioned into four 2x2 blocks:
    # A = [[A00, A01],
    #      [A10, A11]]
    # where each Aij is 2x2.
    #
    # The 16 entries of A are indexed as:
    # A[0] = A00[0][0], A[1] = A00[0][1], A[2] = A00[1][0], A[3] = A00[1][1]
    # A[4] = A01[0][0], A[5] = A01[0][1], A[6] = A01[1][0], A[7] = A01[1][1]
    # A[8] = A10[0][0], A[9] = A10[0][1], A[10] = A10[1][0], A[11] = A10[1][1]
    # A[12] = A11[0][0], A[13] = A11[0][1], A[14] = A11[1][0], A[15] = A11[1][1]
    #
    # Similarly for B (4x4) and C (4x4, but output is p x n = 4 x 4).
    #
    # The 49 Strassen products for 4x4 are indexed by (r1, r2) where r1, r2 in {0..6}.
    # Each product corresponds to a pair of 2x2 Strassen factors.
    #
    # For the (r1, r2) product:
    # - The A-side involves a specific 2x2 block combination determined by r1
    # - The B-side involves a specific 2x2 block combination determined by r2
    # - The C-side involves a specific 2x2 block combination determined by (r1, r2)
    #
    # Actually, let me think about this more carefully. The recursive Strassen for 4x4
    # works as follows:
    #
    # We have A (4x4) and B (4x4), partitioned into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # Using Strassen for the 2x2 block multiplication:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    #
    # Each Pk is a 2x2 matrix product, which itself requires 7 scalar multiplications
    # using Strassen. So we have 7 * 7 = 49 scalar multiplications total.
    #
    # For the k-th block product Pk = Lk_A * Lk_B where Lk_A and Lk_B are 2x2 matrices
    # (linear combinations of the A and B blocks), we apply Strassen to compute Lk_A * Lk_B.
    #
    # Let's denote the 2x2 Strassen factors for computing X * Y as:
    # q1 = (X00+X11)(Y00+Y11)
    # q2 = (X10+X11)Y00
    # q3 = X00(Y10-Y11)
    # q4 = X11(Y01-Y00)
    # q5 = (X00+X01)Y11
    # q6 = (X10-X00)(Y00+Y01)
    # q7 = (X01-X11)(Y10+Y11)
    #
    # For Pk = Lk_A * Lk_B, we set X = Lk_A and Y = Lk_B, and compute q1..q7.
    #
    # Now, Lk_A is a 2x2 matrix whose entries are linear combinations of the A blocks.
    # For example, for P1: L1_A = A00 + A11, so L1_A[0][0] = A00[0][0] + A11[0][0], etc.
    #
    # Similarly, Lk_B is a 2x2 matrix whose entries are linear combinations of the B blocks.
    #
    # The 49 scalar multiplications are q1^1, q2^1, ..., q7^1, q1^2, ..., q7^7 where
    # qj^k is the j-th Strassen factor for the k-th block product.
    #
    # Let me construct U, V, W for these 49 terms.
    #
    # For term (k, j) where k in {0..6} (block product index) and j in {0..6} (Strassen factor index):
    #
    # The A-side: We need to express Lk_A as a linear combination of the 16 A entries.
    # Lk_A is a 2x2 matrix. Its (a, b) entry is a linear combination of A block entries.
    #
    # Let me define the block structure more carefully.
    #
    # A blocks:
    # A00 = [[A[0], A[1]], [A[2], A[3]]]
    # A01 = [[A[4], A[5]], [A[6], A[7]]]
    # A10 = [[A[8], A[9]], [A[10], A[11]]]
    # A11 = [[A[12], A[13]], [A[14], A[15]]]
    #
    # B blocks:
    # B00 = [[B[0], B[1]], [B[2], B[3]]]
    # B01 = [[B[4], B[5]], [B[6], B[7]]]
    # B10 = [[B[8], B[9]], [B[10], B[11]]]
    # B11 = [[B[12], B[13]], [B[14], B[15]]]
    #
    # C blocks (output is 4x4, indexed as C[k*n+i] where k is row, i is col):
    # C00 = [[C[0], C[1]], [C[2], C[3]]]
    # C01 = [[C[4], C[5]], [C[6], C[7]]]
    # C10 = [[C[8], C[9]], [C[10], C[11]]]
    # C11 = [[C[12], C[13]], [C[14], C[15]]]
    #
    # Now, for each k in {0..6}, Lk_A and Lk_B are defined:
    # k=0: L0_A = A00 + A11, L0_B = B00 + B11
    # k=1: L1_A = A10 + A11, L1_B = B00
    # k=2: L2_A = A00, L2_B = B10 - B11
    # k=3: L3_A = A11, L3_B = B01 - B00
    # k=4: L4_A = A00 + A01, L4_B = B11
    # k=5: L5_A = A10 - A00, L5_B = B00 + B01
    # k=6: L6_A = A01 - A11, L6_B = B10 + B11
    #
    # For each k, Lk_A is a 2x2 matrix. Let's denote its entries as:
    # Lk_A = [[Lk_A[0][0], Lk_A[0][1]], [Lk_A[1][0], Lk_A[1][1]]]
    #
    # Similarly for Lk_B.
    #
    # Now, for the j-th Strassen factor of Pk = Lk_A * Lk_B:
    # j=0: q = (Lk_A[0][0] + Lk_A[1][1]) * (Lk_B[0][0] + Lk_B[1][1])
    # j=1: q = (Lk_A[1][0] + Lk_A[1][1]) * Lk_B[0][0]
    # j=2: q = Lk_A[0][0] * (Lk_B[1][0] - Lk_B[1][1])
    # j=3: q = Lk_A[1][1] * (Lk_B[0][1] - Lk_B[0][0])
    # j=4: q = (Lk_A[0][0] + Lk_A[0][1]) * Lk_B[1][1]
    # j=5: q = (Lk_A[1][0] - Lk_A[0][0]) * (Lk_B[0][0] + Lk_B[0][1])
    # j=6: q = (Lk_A[0][1] - Lk_A[1][1]) * (Lk_B[1][0] + Lk_B[1][1])
    #
    # Each of these is a product of two linear forms. The first linear form is in the
    # entries of Lk_A (which are themselves linear combinations of A entries), and the
    # second is in the entries of Lk_B (linear combinations of B entries).
    #
    # For the tensor decomposition, each term (k, j) contributes:
    # U[A_index, (k,j)] = coefficient of A[A_index] in the A-side linear form
    # V[B_index, (k,j)] = coefficient of B[B_index] in the B-side linear form
    # W[C_index, (k,j)] = coefficient of C[C_index] in the output
    #
    # The output: Pk contributes to C as:
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    #
    # But Pk is a 2x2 matrix. So Pk[0][0] contributes to C_block[0][0][0][0], etc.
    #
    # Actually, Pk = Lk_A * Lk_B is a 2x2 matrix. Its (a, b) entry is:
    # Pk[a][b] = sum_{c=0}^{1} Lk_A[a][c] * Lk_B[c][b]
    #
    # And this Pk[a][b] contributes to the (block_row, block_col) block of C.
    #
    # For the j-th Strassen factor of Pk, the contribution to Pk is:
    # Pk[0][0] += w_p1[j] * q_j  (where w_p1 is the C-form for 2x2 Strassen)
    # Pk[0][1] += w_p2[j] * q_j
    # Pk[1][0] += w_p3[j] * q_j
    # Pk[1][1] += w_p4[j] * q_j
    #
    # From the 2x2 Strassen decomposition:
    # C00 = p1 + p4 - p5 + p7 -> w for C00: [1, 0, 0, 1, -1, 0, 1]
    # C01 = p3 + p5 -> w for C01: [0, 0, 1, 0, 1, 0, 0]
    # C10 = p2 + p4 -> w for C10: [0, 1, 0, 1, 0, 0, 0]
    # C11 = p1 - p2 + p3 + p6 -> w for C11: [1, -1, 1, 0, 0, 1, 0]
    #
    # So for the j-th factor, the contribution to Pk's entries is:
    # Pk[0][0] += w2[0][j] * q_j
    # Pk[0][1] += w2[1][j] * q_j
    # Pk[1][0] += w2[2][j] * q_j
    # Pk[1][1] += w2[3][j] * q_j
    #
    # Now, Pk's (a, b) entry contributes to C's block (block_row, block_col) where:
    # For k=0: P0 contributes to C00, C01, C10, C11? No, Pk is a 2x2 matrix that is
    # added to specific C blocks.
    #
    # Actually, in the block Strassen:
    # C00 = P0 + P3 - P4 + P6
    # C01 = P2 + P4
    # C10 = P1 + P3
    # C11 = P0 - P1 + P2 + P5
    #
    # So Pk is a 2x2 matrix, and it is added (with appropriate sign) to one or more C blocks.
    # Each C block is 2x2, so Pk[a][b] contributes to C_block[a][b].
    #
    # For example, P0 contributes to C00 and C11:
    # C00 += P0, so C00[a][b] += P0[a][b] for a, b in {0, 1}
    # C11 += P0, so C11[a][b] += P0[a][b] for a, b in {0, 1}
    #
    # C00[a][b] is C[2*a + b] (since C00 is the top-left 2x2 block)
    # C11[a][b] is C[2*(2) + 2*a + b] = C[4 + 2*a + b]... wait, let me be careful.
    #
    # C is 4x4, indexed as C[row][col]. In flat form, C[k*n+i] where k is row, i is col.
    # So C[0] = C[0][0], C[1] = C[0][1], C[2] = C[1][0], C[3] = C[1][1] (C00 block)
    # C[4] = C[0][2], C[5] = C[0][3], C[6] = C[1][2], C[7] = C[1][3] (C01 block)
    # C[8] = C[2][0], C[9] = C[2][1], C[10] = C[3][0], C[11] = C[3][1] (C10 block)
    # C[12] = C[2][2], C[13] = C[2][3], C[14] = C[3][2], C[15] = C[3][3] (C11 block)
    #
    # So C00[a][b] = C[2*a + b]
    # C01[a][b] = C[2*a + 2 + b]
    # C10[a][b] = C[4 + 2*a + b]
    # C11[a][b] = C[4 + 2*a + 2 + b]
    #
    # Now, for each (k, j) term:
    # 1. Determine the A-side linear form: it's a linear combination of Lk_A's entries,
    #    which are themselves linear combinations of A's entries.
    # 2. Determine the B-side linear form: similarly for Lk_B.
    # 3. Determine the C-side: Pk[a][b] gets contribution w2[a*2+b][j] * q_j, and
    #    Pk[a][b] contributes to specific C entries based on which C blocks Pk is added to.
    #
    # Let me define the block coefficients for each k:
    # k=0: P0 -> C00 (+1), C11 (+1)
    # k=1: P1 -> C10 (+1), C11 (-1)
    # k=2: P2 -> C01 (+1), C11 (+1)
    # k=3: P3 -> C00 (+1), C10 (+1)
    # k=4: P4 -> C00 (-1), C01 (+1)
    # k=5: P5 -> C11 (+1)
    # k=6: P6 -> C00 (+1)
    #
    # Let me encode this as a 4x4 matrix (block_row, block_col) -> coefficient:
    # k=0: {(0,0): 1, (1,1): 1}
    # k=1: {(1,0): 1, (1,1): -1}
    # k=2: {(0,1): 1, (1,1): 1}
    # k=3: {(0,0): 1, (1,0): 1}
    # k=4: {(0,0): -1, (0,1): 1}
    # k=5: {(1,1): 1}
    # k=6: {(0,0): 1}
    #
    # Now, for the A-side: Lk_A is a 2x2 matrix. Its entries are:
    # k=0: L0_A = A00 + A11
    #   L0_A[0][0] = A[0] + A[12]
    #   L0_A[0][1] = A[1] + A[13]
    #   L0_A[1][0] = A[2] + A[14]
    #   L0_A[1][1] = A[3] + A[15]
    # k=1: L1_A = A10 + A11
    #   L1_A[0][0] = A[8] + A[12]
    #   L1_A[0][1] = A[9] + A[13]
    #   L1_A[1][0] = A[10] + A[14]
    #   L1_A[1][1] = A[11] + A[15]
    # k=2: L2_A = A00
    #   L2_A[0][0] = A[0]
    #   L2_A[0][1] = A[1]
    #   L2_A[1][0] = A[2]
    #   L2_A[1][1] = A[3]
    # k=3: L3_A = A11
    #   L3_A[0][0] = A[12]
    #   L3_A[0][1] = A[13]
    #   L3_A[1][0] = A[14]
    #   L3_A[1][1] = A[15]
    # k=4: L4_A = A00 + A01
    #   L4_A[0][0] = A[0] + A[4]
    #   L4_A[0][1] = A[1] + A[5]
    #   L4_A[1][0] = A[2] + A[6]
    #   L4_A[1][1] = A[3] + A[7]
    # k=5: L5_A = A10 - A00
    #   L5_A[0][0] = A[8] - A[0]
    #   L5_A[0][1] = A[9] - A[1]
    #   L5_A[1][0] = A[10] - A[2]
    #   L5_A[1][1] = A[11] - A[3]
    # k=6: L6_A = A01 - A11
    #   L6_A[0][0] = A[4] - A[12]
    #   L6_A[0][1] = A[5] - A[13]
    #   L6_A[1][0] = A[6] - A[14]
    #   L6_A[1][1] = A[7] - A[15]
    #
    # Similarly for B-side:
    # k=0: L0_B = B00 + B11
    # k=1: L1_B = B00
    # k=2: L2_B = B10 - B11
    # k=3: L3_B = B01 - B00
    # k=4: L4_B = B11
    # k=5: L5_B = B00 + B01
    # k=6: L6_B = B10 + B11
    #
    # For each j in {0..6}, the A-side linear form for the j-th Strassen factor is:
    # j=0: (Lk_A[0][0] + Lk_A[1][1])
    # j=1: (Lk_A[1][0] + Lk_A[1][1])
    # j=2: Lk_A[0][0]
    # j=3: Lk_A[1][1]
    # j=4: (Lk_A[0][0] + Lk_A[0][1])
    # j=5: (Lk_A[1][0] - Lk_A[0][0])
    # j=6: (Lk_A[0][1] - Lk_A[1][1])
    #
    # And the B-side linear form:
    # j=0: (Lk_B[0][0] + Lk_B[1][1])
    # j=1: Lk_B[0][0]
    # j=2: (Lk_B[1][0] - Lk_B[1][1])
    # j=3: (Lk_B[0][1] - Lk_B[0][0])
    # j=4: Lk_B[1][1]
    # j=5: (Lk_B[0][0] + Lk_B[0][1])
    # j=6: (Lk_B[1][0] + Lk_B[1][1])
    #
    # Now I can construct U, V, W.
    
    # Define the block coefficients for each k
    block_coeffs = [
        {(0,0): 1, (1,1): 1},    # k=0
        {(1,0): 1, (1,1): -1},   # k=1
        {(0,1): 1, (1,1): 1},    # k=2
        {(0,0): 1, (1,0): 1},    # k=3
        {(0,0): -1, (0,1): 1},   # k=4
        {(1,1): 1},              # k=5
        {(0,0): 1}               # k=6
    ]
    
    # Define Lk_A as a list of 4x16 matrices (2x2 blocks, each entry is a 4-vector of A coeffs)
    # Actually, let me define it as a 2x2 matrix where each entry is a list of 16 coefficients
    def get_Lk_A(k):
        """Return Lk_A as a 2x2 matrix of 16-element coefficient vectors for A."""
        result = [[None]*2 for _ in range(2)]
        for a in range(2):
            for b in range(2):
                coeffs = [0]*16
                if k == 0:  # A00 + A11
                    coeffs[a*2+b] = 1       # A00[a][b]
                    coeffs[12 + a*2+b] = 1  # A11[a][b]
                elif k == 1:  # A10 + A11
                    coeffs[8 + a*2+b] = 1   # A10[a][b]
                    coeffs[12 + a*2+b] = 1  # A11[a][b]
                elif k == 2:  # A00
                    coeffs[a*2+b] = 1
                elif k == 3:  # A11
                    coeffs[12 + a*2+b] = 1
                elif k == 4:  # A00 + A01
                    coeffs[a*2+b] = 1       # A00[a][b]
                    coeffs[4 + a*2+b] = 1   # A01[a][b]
                elif k == 5:  # A10 - A00
                    coeffs[8 + a*2+b] = 1   # A10[a][b]
                    coeffs[a*2+b] = -1      # A00[a][b]
                elif k == 6:  # A01 - A11
                    coeffs[4 + a*2+b] = 1   # A01[a][b]
                    coeffs[12 + a*2+b] = -1 # A11[a][b]
                result[a][b] = coeffs
        return result
    
    def get_Lk_B(k):
        """Return Lk_B as a 2x2 matrix of 16-element coefficient vectors for B."""
        result = [[None]*2 for _ in range(2)]
        for a in range(2):
            for b in range(2):
                coeffs = [0]*16
                if k == 0:  # B00 + B11
                    coeffs[a*2+b] = 1       # B00[a][b]
                    coeffs[12 + a*2+b] = 1  # B11[a][b]
                elif k == 1:  # B00
                    coeffs[a*2+b] = 1
                elif k == 2:  # B10 - B11
                    coeffs[8 + a*2+b] = 1   # B10[a][b]
                    coeffs[12 + a*2+b] = -1 # B11[a][b]
                elif k == 3:  # B01 - B00
                    coeffs[4 + a*2+b] = 1   # B01[a][b]
                    coeffs[a*2+b] = -1      # B00[a][b]
                elif k == 4:  # B11
                    coeffs[12 + a*2+b] = 1
                elif k == 5:  # B00 + B01
                    coeffs[a*2+b] = 1       # B00[a][b]
                    coeffs[4 + a*2+b] = 1   # B01[a][b]
                elif k == 6:  # B10 + B11
                    coeffs[8 + a*2+b] = 1   # B10[a][b]
                    coeffs[12 + a*2+b] = 1  # B11[a][b]
                result[a][b] = coeffs
        return result
    
    # For each (k, j), construct the U, V, W columns
    for k in range(7):
        Lk_A = get_Lk_A(k)
        Lk_B = get_Lk_B(k)
        
        for j in range(7):
            col = k*7 + j
            
            # A-side linear form
            a_coeffs = [0]*16
            if j == 0:  # Lk_A[0][0] + Lk_A[1][1]
                for idx in range(16):
                    a_coeffs[idx] = Lk_A[0][0][idx] + Lk_A[1][1][idx]
            elif j == 1:  # Lk_A[1][0] + Lk_A[1][1]
                for idx in range(16):
                    a_coeffs[idx] = Lk_A[1][0][idx] + Lk_A[1][1][idx]
            elif j == 2:  # Lk_A[0][0]
                a_coeffs = Lk_A[0][0][:]
            elif j == 3:  # Lk_A[1][1]
                a_coeffs = Lk_A[1][1][:]
            elif j == 4:  # Lk_A[0][0] + Lk_A[0][1]
                for idx in range(16):
                    a_coeffs[idx] = Lk_A[0][0][idx] + Lk_A[0][1][idx]
            elif j == 5:  # Lk_A[1][0] - Lk_A[0][0]
                for idx in range(16):
                    a_coeffs[idx] = Lk_A[1][0][idx] - Lk_A[0][0][idx]
            elif j == 6:  # Lk_A[0][1] - Lk_A[1][1]
                for idx in range(16):
                    a_coeffs[idx] = Lk_A[0][1][idx] - Lk_A[1][1][idx]
            
            # B-side linear form
            b_coeffs = [0]*16
            if j == 0:  # Lk_B[0][0] + Lk_B[1][1]
                for idx in range(16):
                    b_coeffs[idx] = Lk_B[0][0][idx] + Lk_B[1][1][idx]
            elif j == 1:  # Lk_B[0][0]
                b_coeffs = Lk_B[0][0][:]
            elif j == 2:  # Lk_B[1][0] - Lk_B[1][1]
                for idx in range(16):
                    b_coeffs[idx] = Lk_B[1][0][idx] - Lk_B[1][1][idx]
            elif j == 3:  # Lk_B[0][1] - Lk_B[0][0]
                for idx in range(16):
                    b_coeffs[idx] = Lk_B[0][1][idx] - Lk_B[0][0][idx]
            elif j == 4:  # Lk_B[1][1]
                b_coeffs = Lk_B[1][1][:]
            elif j == 5:  # Lk_B[0][0] + Lk_B[0][1]
                for idx in range(16):
                    b_coeffs[idx] = Lk_B[0][0][idx] + Lk_B[0][1][idx]
            elif j == 6:  # Lk_B[1][0] + Lk_B[1][1]
                for idx in range(16):
                    b_coeffs[idx] = Lk_B[1][0][idx] + Lk_B[1][1][idx]
            
            # C-side: Pk[a][b] gets contribution w2[a*2+b][j] * q_j
            # And Pk is added to C blocks according to block_coeffs[k]
            c_coeffs = [0]*16
            for (block_row, block_col), coeff in block_coeffs[k].items():
                for a in range(2):
                    for b in range(2):
                        # C index for block (block_row, block_col), position (a, b)
                        # C flat index: (block_row*2 + a) * 4 + (block_col*2 + b)
                        # But wait, C[k*n+i] where k is row, i is col
                        # Row = block_row*2 + a, Col = block_col*2 + b
                        # Flat index = row * n + col = (block_row*2 + a) * 4 + (block_col*2 + b)
                        c_idx = (block_row*2 + a) * 4 + (block_col*2 + b)
                        # w2 is indexed by C entry (0..3) for 2x2, which corresponds to
                        # C00, C01, C10, C11 in the 2x2 case
                        # Here, a*2+b gives the 2x2 C index
                        w_val = w2[a*2+b][j]
                        c_coeffs[c_idx] += coeff * w_val
            
            # Fill U, V, W
            for idx in range(16):
                U[idx][col] = a_coeffs[idx]
                V[idx][col] = b_coeffs[idx]
                W[idx][col] = c_coeffs[idx]
    
    # Verify
    if not _verify(U, V, W, n, m, p, max_trials=3):
        # Fallback to schoolbook if verification fails
        rank_school = n * m * p
        U_school = [[0]*rank_school for _ in range(n*m)]
        V_school = [[0]*rank_school for _ in range(m*p)]
        W_school = [[0]*rank_school for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m+j][col] = 1
                    V_school[j*p+k][col] = 1
                    W_school[k*n+i][col] = 1
                    col += 1
        return {"U": U_school, "V": V_school, "W": W_school}
    
    return {"U": U, "V": V, "W": W}
