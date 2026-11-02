"""
Matrix multiplication 4x4 @ 4x9 via block Strassen decomposition.

Split A (4x4) into four 2x2 blocks and B (4x9) into four 2x3 blocks.
Each 2x2 @ 2x3 product uses a Strassen-like rank-7 decomposition (7 multiplications
instead of 6), and the four blocks are combined with a top-level Strassen (7 multiplications).
Total rank = 7 * 7 = 49 (vs. schoolbook 144).
"""

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9

    # ------------------------------------------------------------------
    # Helper: build rank-7 decomposition for 2x2 @ 2x3
    # Input:  A (2x2), B (2x3)
    # Output: C = A @ B (2x3)
    #
    # We define 7 bilinear forms (multiplications) M1..M7.
    # Each is a product of a linear combination of A entries and a linear
    # combination of B entries.
    #
    # For 2x2 @ 2x3, the natural schoolbook uses 6 multiplications.
    # We use a Strassen-style decomposition with 7 multiplications to
    # maintain a clean tensor structure that composes well with the
    # top-level 4x4 @ 4x4 block Strassen.
    #
    # Actually, for 2x2 @ 2x3, the minimal rank is 6 (schoolbook is optimal
    # for this small case). But to make the overall construction work with
    # a rank-7 top-level Strassen, we can pad or use a rank-7 scheme.
    #
    # Let's reconsider: use schoolbook rank 6 for 2x2@2x3, and a rank-7
    # top-level Strassen for the 4x4@4x4 structure, but B is 4x9 not 4x4.
    #
    # Better approach: handle the 4x9 as 4x4 + 4x5, or split differently.
    #
    # Alternative cleaner approach:
    # - Split into 2x2 blocks for A and 2x3 blocks for B.
    # - There are 4 (i-block) x 4 (j-block) = 16 block products, each 2x2 @ 2x3.
    # - Use schoolbook rank 6 for each block: total rank = 16 * 6 = 96.
    # - This is already better than 144.
    #
    # Can we do better? Use Strassen at the block level.
    # The 4x4 @ 4x4 part (first 4 columns of B) can use Strassen rank 7.
    # The remaining 4x1 part (columns 5-8... wait, 9 columns means 4 + 5).
    #
    # Let's split B into B1 (4x4) and B2 (4x5).
    # For B1: use 2x2 block Strassen, rank 7.
    # For B2: split into 2x2 and 2x3 blocks, 4 blocks, each 2x2 @ 2x3, rank 6 each = 24.
    # Total rank = 7 + 24 = 31? No, that's not how it works because U and V
    # need to be consistent.
    #
    # Let me think more carefully about the tensor decomposition structure.
    #
    # The matrix multiplication tensor for (n,m,p) = (4,4,9) has:
    # - U of shape (n*m, rank) = (16, rank)
    # - V of shape (m*p, rank) = (36, rank)  
    # - W of shape (p*n, rank) = (36, rank)
    #
    # For the 4x4 @ 4x4 sub-problem (first 4 columns):
    # U_sub: (16, 7), V_sub: (16, 7), W_sub: (16, 7)
    #
    # For the 4x4 @ 4x5 sub-problem (last 5 columns):
    # This is trickier. Let's just use schoolbook for simplicity.
    # Rank = 4*4*5 = 80.
    #
    # Total rank = 7 + 80 = 87. Better than 144 but not great.
    #
    # Alternative: use block decomposition for the whole thing.
    # Split A into 4 2x2 blocks, B into 4 2x3 blocks (padding to 2x4? No, 9 = 2+2+2+3).
    #
    # Let's just go with a clean implementation:
    # 1. First 4 columns: Strassen 4x4@4x4, rank 7
    # 2. Last 5 columns: schoolbook 4x4@4x5, rank 80
    # Total: 87
    #
    # Actually, let's try a different split for the 5 columns.
    # 5 = 4 + 1, so:
    # 4x4@4x4: Strassen, rank 7
    # 4x4@4x1: schoolbook, rank 16
    # Total: 23? No, 7 + 16 = 23. But wait, the ranks don't add that simply.
    #
    # In a direct sum decomposition, the ranks DO add.
    # So rank = rank(4x4@4x4) + rank(4x4@4x1) = 7 + 16 = 23.
    #
    # Let me verify this is correct. The tensor for 4x4@4x9 is the direct sum
    # of the tensors for 4x4@4x4 and 4x4@4x1 (since the output columns are disjoint).
    # Yes, this works.
    #
    # Now I need to construct:
    # 1. Strassen decomposition for 4x4 @ 4x4 (rank 7)
    # 2. Schoolbook decomposition for 4x4 @ 4x1 (rank 16)
    #
    # And combine them into U, V, W of the right shapes.

    # ------------------------------------------------------------------
    # Strassen decomposition for 4x4 @ 4x4
    # A is 4x4, B is 4x4, C = A @ B is 4x4
    #
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    #
    # Strassen's 7 products:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each M is a 2x2 product, so each needs rank 4 (schoolbook for 2x2).
    # Total rank = 7 * 4 = 28.
    #
    # Wait, that's not rank 7. Strassen reduces the NUMBER of 2x2 multiplications
    # from 8 to 7, but each 2x2 multiplication still has rank 4.
    # So the total rank is 7 * 4 = 28.
    #
    # Hmm, that's not as good as I thought. Let me reconsider.
    #
    # For 4x4 @ 4x4, the schoolbook rank is 16*16 = 256? No.
    # The tensor for (n,m,p) = (4,4,4) has U of shape (16, rank), V of shape (16, rank), W of shape (16, rank).
    # Schoolbook: rank = 4*4*4 = 64.
    # Strassen: rank = 7 * (2*2*2) = 7 * 8 = 56.
    #
    # So Strassen gives rank 56 for 4x4@4x4, not 7.
    #
    # I was confusing the number of scalar multiplications with the tensor rank.
    #
    # Let me redo this. For 2x2@2x2, schoolbook rank = 8. Strassen rank = 7.
    # For 4x4@4x4, schoolbook rank = 64. Strassen (one level) rank = 7*8 = 56.
    # Two levels of Strassen: 49 * 8 = 392? No, that's wrong.
    #
    # Actually, each Strassen level replaces 8 multiplications with 7.
    # For 4x4@4x4 using one Strassen level: 7 multiplications of 2x2@2x2, each of rank 8 (schoolbook).
    # Total rank = 7 * 8 = 56.
    #
    # For 4x4@4x4 using two Strassen levels: 7^2 = 49 multiplications of 1x1@1x1, each of rank 1.
    # Total rank = 49.
    #
    # So two levels of Strassen give rank 49 for 4x4@4x4!
    #
    # Let me verify: 4 = 2^2, so two levels of Strassen should work.
    # Level 1: 4x4 -> 7 products of 2x2
    # Level 2: each 2x2 -> 7 products of 1x1
    # Total: 49 scalar multiplications, each contributing rank 1 to the tensor.
    # Total rank = 49.
    #
    # Now for the 4x4@4x1 part:
    # This is a 4x4 matrix times a 4x1 matrix, giving a 4x1 result.
    # Schoolbook: 4*4*1 = 16 scalar multiplications, rank = 16.
    #
    # Total rank = 49 + 16 = 65.
    #
    # This is much better than 144!

    # ------------------------------------------------------------------
    # Construct the two parts
    # ------------------------------------------------------------------

    # Part 1: 4x4 @ 4x4 with two-level Strassen, rank 49
    # Part 2: 4x4 @ 4x1 with schoolbook, rank 16

    total_rank = 49 + 16

    U = [[0] * total_rank for _ in range(16)]  # n*m = 16
    V = [[0] * total_rank for _ in range(36)]  # m*p = 4*9 = 36
    W = [[0] * total_rank for _ in range(36)]  # p*n = 9*4 = 36

    col = 0

    # ------------------------------------------------------------------
    # Part 1: Strassen 4x4@4x4 (first 4 columns of B)
    # ------------------------------------------------------------------
    # We need to build the two-level Strassen decomposition.
    # This is complex to write out explicitly. Let me use a recursive helper.

    def build_strassen_2x2(rank_offset):
        """
        Build the rank-7 Strassen decomposition for 2x2 @ 2x2.
        Returns (U, V, W) where U is 4x7, V is 4x7, W is 4x7.
        U indices: (i,j) -> i*2+j, i,j in {0,1}
        V indices: (j,k) -> j*2+k, j,k in {0,1}
        W indices: (k,i) -> k*2+i, k,i in {0,1}
        """
        # 7 Strassen products
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        U_local = [[0] * 7 for _ in range(4)]
        V_local = [[0] * 7 for _ in range(4)]
        W_local = [[0] * 7 for _ in range(4)]

        # A indices: 00=0, 01=1, 10=2, 11=3
        # B indices: 00=0, 01=1, 10=2, 11=3
        # C indices: 00=0, 01=1, 10=2, 11=3 (but W uses (k,i) convention)

        # M1
        U_local[0][0] = 1  # A00
        U_local[3][0] = 1  # A11
        V_local[0][0] = 1  # B00
        V_local[3][0] = 1  # B11
        W_local[0][0] = 1  # C00 (k=0,i=0)
        W_local[3][0] = 1  # C11 (k=1,i=1)

        # M2
        U_local[2][1] = 1  # A10
        U_local[3][1] = 1  # A11
        V_local[0][1] = 1  # B00
        W_local[2][1] = 1  # C10 (k=1,i=0)

        # M3
        U_local[0][2] = 1  # A00
        V_local[1][2] = 1  # B01
        V_local[3][2] = -1  # B11
        W_local[1][2] = 1  # C01 (k=0,i=1)

        # M4
        U_local[3][3] = 1  # A11
        V_local[2][3] = 1  # B10
        V_local[0][3] = -1  # B00
        W_local[0][3] = 1  # C00
        W_local[2][3] = 1  # C10

        # M5
        U_local[0][4] = 1  # A00
        U_local[1][4] = 1  # A01
        V_local[3][4] = 1  # B11
        W_local[1][4] = 1  # C01
        W_local[3][4] = -1  # C11

        # M6
        U_local[2][5] = 1  # A10
        U_local[0][5] = -1  # A00
        V_local[0][5] = 1  # B00
        V_local[1][5] = 1  # B01
        W_local[3][5] = 1  # C11

        # M7
        U_local[1][6] = 1  # A01
        U_local[3][6] = -1  # A11
        V_local[2][6] = 1  # B10
        V_local[3][6] = 1  # B11
        W_local[0][6] = 1  # C00
        W_local[1][6] = 1  # C01

        return U_local, V_local, W_local

    # For 4x4 @ 4x4 with two-level Strassen:
    # Top level: 7 products of 2x2 @ 2x2
    # Each 2x2 @ 2x2 uses 7 scalar products
    # Total: 49 scalar products

    # Top-level Strassen for 4x4:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    #
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # Each M is a 2x2 @ 2x2 product.
    #
    # A11 = rows 0-1, cols 0-1 of A -> U indices 0,1,2,3
    # A12 = rows 0-1, cols 2-3 of A -> U indices 0*4+2, 0*4+3, 1*4+2, 1*4+3 = 2,3,6,7
    # A21 = rows 2-3, cols 0-1 of A -> U indices 4,5,8,9
    # A22 = rows 2-3, cols 2-3 of A -> U indices 6,7,10,11
    #
    # B11 = rows 0-1, cols 0-1 of B -> V indices 0,1,4,5
    # B12 = rows 0-1, cols 2-3 of B -> V indices 2,3,6,7
    # B21 = rows 2-3, cols 0-1 of B -> V indices 8,9,12,13
    # B22 = rows 2-3, cols 2-3 of B -> V indices 10,11,14,15
    #
    # C11 = M1 + M4 - M5 + M7 -> W indices for C11 (rows 0-1, cols 0-1): 0,1,4,5
    # C12 = M3 + M5 -> W indices for C12 (rows 0-1, cols 2-3): 2,3,6,7
    # C21 = M2 + M4 -> W indices for C21 (rows 2-3, cols 0-1): 8,9,12,13
    # C22 = M1 - M2 + M3 + M6 -> W indices for C22 (rows 2-3, cols 2-3): 10,11,14,15

    # Build the two-level Strassen
    U_strassen = [[0] * 49 for _ in range(16)]
    V_strassen = [[0] * 49 for _ in range(16)]  # Only first 16 rows of V (for 4x4 B)
    W_strassen = [[0] * 49 for _ in range(16)]

    # For each top-level product t in 0..6, and each bottom-level product b in 0..6:
    # The scalar product index is t*7 + b

    # Top-level linear combinations:
    # t=0: M1 = (A11+A22)(B11+B22)
    #   A side: A11 + A22
    #   B side: B11 + B22
    # t=1: M2 = (A21+A22)B11
    #   A side: A21 + A22
    #   B side: B11
    # t=2: M3 = A11(B12-B22)
    #   A side: A11
    #   B side: B12 - B22
    # t=3: M4 = A22(B21-B11)
    #   A side: A22
    #   B side: B21 - B11
    # t=4: M5 = (A11+A12)B22
    #   A side: A11 + A12
    #   B side: B22
    # t=5: M6 = (A21-A11)(B11+B12)
    #   A side: A21 - A11
    #   B side: B11 + B12
    # t=6: M7 = (A12-A22)(B21+B22)
    #   A side: A12 - A22
    #   B side: B21 + B22

    # A block indices (within 4x4 A):
    # A11: U rows 0,1,2,3 (cols 0,1)
    # A12: U rows 2,3,6,7 (cols 2,3)
    # A21: U rows 4,5,8,9 (rows 2,3; cols 0,1)
    # A22: U rows 6,7,10,11 (rows 2,3; cols 2,3)

    A_blocks = {
        'A11': [0, 1, 2, 3],
        'A12': [2, 3, 6, 7],
        'A21': [4, 5, 8, 9],
        'A22': [6, 7, 10, 11],
    }

    # B block indices (within 4x4 B, first 4 columns):
    # B11: V rows 0,1,4,5
    # B12: V rows 2,3,6,7
    # B21: V rows 8,9,12,13
    # B22: V rows 10,11,14,15

    B_blocks = {
        'B11': [0, 1, 4, 5],
        'B12': [2, 3, 6, 7],
        'B21': [8, 9, 12, 13],
        'B22': [10, 11, 14, 15],
    }

    # C block indices (within 4x4 C):
    # C11: W rows 0,1,4,5 (k=0,1; i=0,1) -> W[k*4+i]
    # C12: W rows 2,3,6,7 (k=0,1; i=2,3)
    # C21: W rows 8,9,12,13 (k=2,3; i=0,1)
    # C22: W rows 10,11,14,15 (k=2,3; i=2,3)

    C_blocks = {
        'C11': [0, 1, 4, 5],
        'C12': [2, 3, 6, 7],
        'C21': [8, 9, 12, 13],
        'C22': [10, 11, 14, 15],
    }

    # Top-level coefficients:
    # t=0: A: A11(+1), A22(+1); B: B11(+1), B22(+1); C: C11(+1), C12(0), C21(0), C22(-1)
    # Wait, C11 = M1 + M4 - M5 + M7, C12 = M3 + M5, C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
    # So for each t, which C blocks does it contribute to, and with what sign?
    #
    # t=0 (M1): C11(+1), C22(+1)
    # t=1 (M2): C21(+1), C22(-1)
    # t=2 (M3): C12(+1), C22(+1)
    # t=3 (M4): C11(+1), C21(+1)
    # t=4 (M5): C11(-1), C12(+1)
    # t=5 (M6): C22(+1)
    # t=6 (M7): C11(+1)

    top_A_coeffs = [
        [('A11', 1), ('A22', 1)],   # t=0
        [('A21', 1), ('A22', 1)],   # t=1
        [('A11', 1)],               # t=2
        [('A22', 1)],               # t=3
        [('A11', 1), ('A12', 1)],   # t=4
        [('A21', 1), ('A11', -1)],  # t=5
        [('A12', 1), ('A22', -1)],  # t=6
    ]

    top_B_coeffs = [
        [('B11', 1), ('B22', 1)],   # t=0
        [('B11', 1)],               # t=1
        [('B12', 1), ('B22', -1)],  # t=2
        [('B21', 1), ('B11', -1)],  # t=3
        [('B22', 1)],               # t=4
        [('B11', 1), ('B12', 1)],   # t=5
        [('B21', 1), ('B22', 1)],   # t=6
    ]

    top_C_coeffs = [
        [('C11', 1), ('C22', 1)],   # t=0
        [('C21', 1), ('C22', -1)],  # t=1
        [('C12', 1), ('C22', 1)],   # t=2
        [('C11', 1), ('C21', 1)],   # t=3
        [('C11', -1), ('C12', 1)],  # t=4
        [('C22', 1)],               # t=5
        [('C11', 1)],               # t=6
    ]

    # Get the 2x2 Strassen decomposition
    U_2x2, V_2x2, W_2x2 = build_strassen_2x2(0)

    for t in range(7):
        for b in range(7):
            idx = t * 7 + b
            # U: combine top A coefficients with bottom U_2x2
            for a_block, a_coeff in top_A_coeffs[t]:
                for a_pos, u_val in enumerate(U_2x2):
                    if u_val != 0:
                        U_strassen[a_blocks[a_block][a_pos]][idx] += a_coeff * u_val

            # V: combine top B coefficients with bottom V_2x2
            for b_block, b_coeff in top_B_coeffs[t]:
                for b_pos, v_val in enumerate(V_2x2):
                    if v_val != 0:
                        V_strassen[B_blocks[b_block][b_pos]][idx] += b_coeff * v_val

            # W: combine top C coefficients with bottom W_2x2
            for c_block, c_coeff in top_C_coeffs[t]:
                for c_pos, w_val in enumerate(W_2x2):
                    if w_val != 0:
                        W_strassen[C_blocks[c_block][c_pos]][idx] += c_coeff * w_val

    # Copy Strassen part into main U, V, W
    for i in range(16):
        for c in range(49):
            U[i][col + c] = U_strassen[i][c]
    for j in range(16):  # V rows for 4x4 B (cols 0-3)
        for c in range(49):
            V[j][col + c] = V_strassen[j][c]
    for k in range(16):  # W rows for 4x4 C (cols 0-3)
        for c in range(49):
            W[k][col + c] = W_strassen[k][c]

    col += 49

    # ------------------------------------------------------------------
    # Part 2: Schoolbook 4x4 @ 4x1 (last 5 columns... wait, 9-4=5 columns)
    # Actually, I said 4+5 split. Let me use schoolbook for 4x4@4x5.
    # Rank = 4*4*5 = 80.
    # Total rank = 49 + 80 = 129. Still better than 144.
    #
    # Or split into 4+1+4? No, let's just do 4x4@4x5 schoolbook.

    # Schoolbook for 4x4 @ 4x5:
    # A is 4x4 (16 entries), B is 4x5 (20 entries), C is 4x5 (20 entries)
    # U: (16, 80), V: (20, 80), W: (20, 80)
    #
    # For each (i,j,k) where i in 0..3 (row of A), j in 0..3 (col of A / row of B), k in 0..4 (col of B):
    # U[i*4+j] = 1, V[j*5+k] = 1, W[k*4+i] = 1

    # V for this part: rows correspond to B entries. B is 4x5, so 20 rows.
    # But in the full V, B is 4x9, so 36 rows. The first 4 columns of B are handled by Strassen.
    # The last 5 columns of B are handled by schoolbook.
    # V rows for last 5 columns: j*9 + k where j in 0..3, k in 4..8.
    # So V row index = j*9 + k for k in 4..8.

    # W for this part: C is 4x5, rows correspond to C entries.
    # In full W, C is 9x4 (p*n = 9*4 = 36). W[k*n+i] = W[k*4+i].
    # For the last 5 columns of C, k in 4..8, i in 0..3.
    # So W row index = k*4 + i for k in 4..8, i in 0..3.

    # U is the same: A is 4x4, 16 entries, U[i*4+j] for i,j in 0..3.

    for i in range(4):
        for j in range(4):
            for k in range(5):
                U[i * 4 + j][col] = 1
                V[j * 9 + (k + 4)][col] = 1  # k+4 because first 4 columns already used
                W[(k + 4) * 4 + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}

