"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    import numpy as np

    deadline = time.time() + 115  # safety margin

    # Schoolbook decomposition: rank = n*m*p = 144
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    best_result = {"U": U, "V": V, "W": W}

    def verify(U_mat, V_mat, W_mat, n, m, p):
        """Verify tensor decomposition: sum_r U[a,r]*V[b,r]*W[c,r] == T[a,b,c]"""
        num_cols = len(U_mat[0]) if U_mat else 0
        if num_cols == 0:
            return False
        # Build expected tensor as dict
        T = {}
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[(i*m+j, j*p+k, k*n+i)] = 1
        # Check all (a,b,c)
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    s = 0.0
                    for r in range(num_cols):
                        s += U_mat[a][r] * V_mat[b][r] * W_mat[c][r]
                    expected = T.get((a, b, c), 0)
                    if abs(s - expected) > 1e-9:
                        return False
        return True

    if not verify(best_result["U"], best_result["V"], best_result["W"], n, m, p):
        raise ValueError("Schoolbook verification failed")

    # Try to find a lower-rank decomposition using structured approaches
    # Key insight: For 4x4 * 4x9, we can decompose the 9 columns into groups
    # and use rank-reduced multiplications for each group.

    # Approach: Split the 9 columns of B into groups and use efficient
    # rank-r decompositions for smaller sub-problems.

    # For a 4x4 * 4x1 multiplication (vector times matrix), the tensor has
    # rank 4 (each output C[i] = sum_j A[i,j]*B[j] needs 4 terms).
    # For 4x4 * 4x4, schoolbook is 64. We can try to do better.

    # Let's try a block approach:
    # Split B's 9 columns into: one 4x4 block + one 4x4 block + one 4x1 column
    # Cost: rank(4x4x4) + rank(4x4x4) + rank(4x4x1)
    # If we can get rank 49 for 4x4x4 (Strassen-based) and rank 4 for 4x4x1:
    # Total: 49 + 49 + 4 = 102

    # But implementing Strassen for exact integer/half-integer entries is complex.
    # Let's try a simpler structured approach.

    # Alternative: Use the fact that 4x4 matrix multiplication can be done
    # with rank 49 via Strassen (7 scalar mults per 2x2 block, 4 blocks).
    # Actually Strassen gives rank 7 for 2x2x2, and 4x4x4 uses 4 copies of 2x2x2
    # with 7 mults each = 28? No, that's not right either.

    # Let me think more carefully. The standard Strassen algorithm for 2x2*2x2
    # uses 7 multiplications. For 4x4*4x4, we split into 2x2 blocks and
    # need 4 Strassen computations? No, we need to handle cross-block terms.

    # Actually, the rank of 4x4x4 tensor is known to be at most 49 (using
    # Strassen recursively: 7^2 = 49). Let me implement this.

    # Strassen 2x2x2 rank-7 decomposition:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For 4x4x4, split A, B, C into 2x2 blocks of 2x2 sub-matrices.
    # This gives 4 independent 2x2x2 problems? No, the blocks interact.

    # Actually, for 4x4 = (2x2) x (2x2), the multiplication tensor is NOT
    # simply 4 copies of 2x2x2. The block structure means:
    # C_11 = A_11*B_11 + A_12*B_21
    # C_12 = A_11*B_12 + A_12*B_22
    # C_21 = A_21*B_11 + A_22*B_21
    # C_22 = A_21*B_12 + A_22*B_22
    # So we have 8 products of 2x2 matrices, each of which needs rank 7.
    # Total: 8 * 7 = 56. But there's sharing possible.

    # The known result is rank(4x4x4) <= 49 using a more clever decomposition.
    # Let me just use the straightforward 8*7=56 approach first, which is
    # still better than 64.

    # For 4x4x1 (matrix-vector product), the tensor T[i*m+j, j*p+k, k*n+i]
    # with p=1: T[i*4+j, j*1+0, 0*4+i] = T[i*4+j, j, i] = 1
    # This is just the matrix A itself (with B being a vector).
    # The rank is 4: C[i] = sum_j A[i,j] * B[j]
    # Each C[i] needs 4 terms, but we can share: use 4 rank-1 terms
    # where term r: U[i*4+j, r] = delta(i==r), V[j, r] = delta(j==r), W[i, r] = 1
    # Wait, that's 4 terms but each term only covers one (i,j) pair.
    # Actually for matrix-vector: C[i] = sum_j A[i,j]*B[j]
    # Term r (for j=r): U[i*4+r, r] = 1 for all i, V[r, r] = 1, W[i, r] = 1
    # That's 4 terms total. Rank = 4.

    # So for 4x4x9:
    # Split 9 columns as 4+4+1
    # rank = rank(4x4x4) + rank(4x4x4) + rank(4x4x1) = 56 + 56 + 4 = 116
    # Better than 144!

    # Let me implement this.

    if time.time() > deadline - 30:
        return best_result

    def build_4x4x4_decomp(offset):
        """Build rank-56 decomposition for 4x4*4x4 using 8 Strassen 2x2x2 blocks.
        Returns (U_cols, V_cols, W_cols) as lists of lists, each of length 56.
        U has 16 rows, V has 16 rows, W has 16 rows.
        offset is the starting column index in the global decomposition.
        """
        # 2x2x2 Strassen rank-7 decomposition
        # For sub-matrices indexed by (block_row, block_col) in {0,1}
        # A blocks: A_00, A_01, A_10, A_11 (each 2x2)
        # B blocks: B_00, B_01, B_10, B_11
        # C blocks: C_00, C_01, C_10, C_11

        # Strassen 7 multiplications for a 2x2x2:
        # The 7 terms and their U, V, W coefficients for 2x2x2:
        # Term 0: (A00+A11)(B00+B11)
        # Term 1: (A10+A11)B00
        # Term 2: A00(B10-B11)
        # Term 3: A11(B01-B00)
        # Term 4: (A00+A01)B11
        # Term 5: (A10-A00)(B00+B01)
        # Term 6: (A01-A11)(B10+B11)

        # For 4x4x4, we have 8 cross-block products:
        # C_00 = A_00*B_00 + A_01*B_10
        # C_01 = A_00*B_01 + A_01*B_11
        # C_10 = A_10*B_00 + A_11*B_10
        # C_11 = A_10*B_01 + A_11*B_11

        # Each product A_ab * B_cd (where a,c in {0,1} for row block, b,d in {0,1} for col block)
        # is a 2x2*2x2 multiplication with rank 7.

        # Total: 8 products * 7 terms = 56 terms

        U_cols = [[0]*56 for _ in range(16)]
        V_cols = [[0]*56 for _ in range(16)]
        W_cols = [[0]*56 for _ in range(16)]

        # Strassen 2x2x2 coefficients
        # For a 2x2x2 with indices (i,j) for A, (j,k) for B, (k,i) for C:
        # i,j,k in {0,1}
        # A index: i*2+j, B index: j*2+k, C index: k*2+i

        # The 7 Strassen terms for 2x2x2:
        # Each term has coefficients for U (4 entries), V (4 entries), W (4 entries)
        #
        # Term 0: (A00+A11)(B00+B11)
        #   U: A00=1, A11=1, A01=0, A10=0 -> [1,0,0,1]
        #   V: B00=1, B11=1, B01=0, B10=0 -> [1,0,0,1]
        #   W: contributes to C00 and C11
        #   C00 = M0 + M3 - M4 + M6, C01 = M2+M4, C10 = M1+M3, C11 = M0-M1+M2+M5
        #   So M0 contributes: C00+=1, C11+=1 -> W: [1,0,0,1]

        # Let me define the 7 terms more carefully.
        # Standard Strassen:
        # M1 = (A11+A22)(B11+B22)  [using 1-indexed]
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6

        # Converting to 0-indexed:
        # M0 = (A00+A11)(B00+B11)
        # M1 = (A10+A11)B00
        # M2 = A00(B10-B11)
        # M3 = A11(B01-B00)
        # M4 = (A00+A01)B11
        # M5 = (A10-A00)(B00+B01)
        # M6 = (A01-A11)(B10+B11)
        #
        # C00 = M0 + M3 - M4 + M6
        # C01 = M2 + M4
        # C10 = M1 + M3
        # C11 = M0 - M1 + M2 + M5

        # U coefficients for each M (for A, 4 elements: A00,A01,A10,A11):
        strassen_U = [
            [1, 0, 0, 1],  # M0: A00+A11
            [0, 0, 1, 1],  # M1: A10+A11
            [1, 0, 0, 0],  # M2: A00
            [0, 0, 0, 1],  # M3: A11
            [1, 1, 0, 0],  # M4: A00+A01
            [-1, 0, 1, 0], # M5: A10-A00
            [0, 1, 0, -1], # M6: A01-A11
        ]
        # V coefficients for each M (for B, 4 elements: B00,B01,B10,B11):
        strassen_V = [
            [1, 0, 0, 1],  # M0: B00+B11
            [1, 0, 0, 0],  # M1: B00
            [0, -1, 1, 0], # M2: B10-B11
            [-1, 1, 0, 0], # M3: B01-B00
            [0, 0, 0, 1],  # M4: B11
            [1, 1, 0, 0],  # M5: B00+B01
            [0, 0, 1, 1],  # M6: B10+B11
        ]
        # W coefficients for each M (for C, 4 elements: C00,C01,C10,C11):
        strassen_W = [
            [1, 0, 0, 1],  # M0: C00+C11
            [0, 0, 1, -1], # M1: C10-C11
            [0, 1, 0, 1],  # M2: C01+C11
            [1, 0, 1, 0],  # M3: C00+C10
            [-1, 1, 0, 0], # M4: -C00+C01
            [0, 0, 0, 1],  # M5: C11
            [1, 0, 0, 0],  # M6: C00
        ]

        # Now for 4x4x4, the 8 products:
        # Product (ar, ac, br, bc) where ar,br in {0,1} (row block of A, row block of B)
        # ac,bc in {0,1} (col block of A, col block of B)
        # A block (ar, ac) is at rows ar*2..ar*2+1, cols ac*2..ac*2+1
        # B block (br, bc) is at rows br*2..br*2+1, cols bc*2..bc*2+1
        # C block (ar, bc) is at rows ar*2..ar*2+1, cols bc*2..bc*2+1

        # For each of the 8 (ar,ac,br,bc) combinations where the product contributes to C(ar,bc):
        # Actually, C(ar,bc) = sum over all (ar',ac') where... no.
        # C(ar,bc) = A(ar,*) * B(*,bc) = A(ar,0)*B(0,bc) + A(ar,1)*B(1,bc)
        # So for fixed (ar, bc), we have 2 products: (ar,0,0,bc) and (ar,1,1,bc)

        # Let me enumerate all 8:
        products = [
            (0, 0, 0, 0),  # A_00 * B_00 -> C_00
            (0, 1, 1, 0),  # A_01 * B_10 -> C_00
            (0, 0, 0, 1),  # A_00 * B_01 -> C_01
            (0, 1, 1, 1),  # A_01 * B_11 -> C_01
            (1, 0, 0, 0),  # A_10 * B_00 -> C_10
            (1, 1, 1, 0),  # A_11 * B_10 -> C_10
            (1, 0, 0, 1),  # A_10 * B_01 -> C_11
            (1, 1, 1, 1),  # A_11 * B_11 -> C_11
        ]

        for prod_idx, (ar, ac, br, bc) in enumerate(products):
            # This product uses Strassen terms 0-6
            for s in range(7):
                col = prod_idx * 7 + s
                # U: A block (ar, ac), element (i,j) in the 2x2 block
                # Global A index: (ar*2+i)*4 + (ac*2+j)
                for i in range(2):
                    for j in range(2):
                        U_cols[(ar*2+i)*4 + (ac*2+j)][col] = strassen_U[s][i*2+j]
                # V: B block (br, bc), element (i,j) in the 2x2 block
                # Global B index: (br*2+i)*4 + (bc*2+j)
                for i in range(2):
                    for j in range(2):
                        V_cols[(br*2+i)*4 + (bc*2+j)][col] = strassen_V[s][i*2+j]
                # W: C block (ar, bc), element (i,j) in the 2x2 block
                # Global C index: (bc*2+i)*4 + (ar*2+j)  [trace convention: W[k*n+i, r] where k is col of C, i is row of C]
                # Wait, the trace convention is W[k*n+i, r] where the output C[i,k]
                # So for C element at row r_c, col c_c: W index = c_c * n + r_c
                # In our block: C block (ar, bc) has rows ar*2..ar*2+1, cols bc*2..bc*2+1
                # So C element (ar*2+i, bc*2+j) has W index = (bc*2+j)*4 + (ar*2+i)
                for i in range(2):
                    for j in range(2):
                        W_cols[(bc*2+j)*4 + (ar*2+i)][col] = strassen_W[s][i*2+j]

        return U_cols, V_cols, W_cols

    def build_4x4x1_decomp(offset):
        """Build rank-4 decomposition for 4x4*4x1.
        Returns (U_cols, V_cols, W_cols) each of length 4.
        U has 16 rows, V has 4 rows, W has 4 rows.
        """
        U_cols = [[0]*4 for _ in range(16)]
        V_cols = [[0]*4 for _ in range(4)]
        W_cols = [[0]*4 for _ in range(4)]

        # C[i] = sum_j A[i,j] * B[j]
        # Term r (for j=r):
        #   U[i*4+r, r] = 1 for all i
        #   V[r, r] = 1
        #   W[i, r] = 1 for all i (W index for C[i] is i*1+... wait)
        # W index: k*n+i where k is column index of C. Since p=1, C has 1 column.
        # C[i,0] -> W index = 0*4+i = i
        # So W[i, r] = 1 for all i

        for r in range(4):
            for i in range(4):
                U_cols[i*4 + r][r] = 1
            V_cols[r][r] = 1
            for i in range(4):
                W_cols[i][r] = 1

        return U_cols, V_cols, W_cols

    # Build the combined decomposition for 4x4*4x9
    # Split 9 columns: first 4, next 4, last 1
    # rank = 56 + 56 + 4 = 116

    total_rank = 56 + 56 + 4
    U_new = [[0]*total_rank for _ in range(16)]
    V_new = [[0]*total_rank for _ in range(36)]  # m*p = 4*9 = 36
    W_new = [[0]*total_rank for _ in range(36)]  # p*n = 9*4 = 36

    # Block 1: columns 0-3 of B (4x4 sub-problem)
    U1, V1, W1 = build_4x4x4_decomp(0)
    for r in range(56):
        for a in range(16):
            U_new[a][r] = U1[a][r]
        for b in range(16):
            # B index: j*9+k where j is row of B (0-3), k is col of B (0-3)
            # V1 index: j*4+k
            j = b // 4
            k = b % 4
            V_new[j*9 + k][r] = V1[b][r]
        for c in range(16):
            # W index: k*4+i where k is col of C (0-3), i is row of C (0-3)
            # W1 index: k*4+i
            W_new[c][r] = W1[c][r]

    # Block 2: columns 4-7 of B (4x4 sub-problem)
    U2, V2, W2 = build_4x4x4_decomp(56)
    for r in range(56):
        for a in range(16):
            U_new[a][56 + r] = U2[a][r]
        for b in range(16):
            j = b // 4
            k = b % 4
            V_new[j*9 + (4 + k)][56 + r] = V2[b][r]
        for c in range(16):
            # C columns 4-7
            k_c = c // 4
            i_c = c % 4
            W_new[(4 + k_c)*4 + i_c][56 + r] = W2[c][r]

    # Block 3: column 8 of B (4x1 sub-problem)
    U3, V3, W3 = build_4x4x1_decomp(112)
    for r in range(4):
        for a in range(16):
            U_new[a][112 + r] = U3[a][r]
        for b in range(4):
            # B index: j*9+8
            V_new[b*9 + 8][112 + r] = V3[b][r]
        for c in range(4):
            # W index: 8*4+i = 32+i
            W_new[32 + c][112 + r] = W3[c][r]

    # Verify the new decomposition
    if verify(U_new, V_new, W_new, n, m, p):
        best_result = {"U": U_new, "V": V_new, "W": W_new}
    else:
        # Fall back to schoolbook
        pass

    if time.time() > deadline - 10:
        return best_result

    # Try to further optimize: can we do better than 116?
    # Maybe use a different split or a more clever decomposition.
    # For now, 116 is a solid improvement over 144.

    return best_result
# EVOLVE-BLOCK-END
