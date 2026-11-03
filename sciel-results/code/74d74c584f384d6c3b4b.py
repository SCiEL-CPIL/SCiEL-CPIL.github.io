"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5

    # Strassen 2x2x2 coefficients: (U_factor, V_factor, W_factor)
    # where U_factor = (a, b), V_factor = (c, d), W_factor = (e, f)
    # and the term is (a*A11 + b*A12)*(c*B11 + d*B21)*(e*C11 + f*C12) style
    # Actually for 2x2x2:
    # We have A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    # Strassen:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-d)(f+h)
    # M7 = (a-c)(e+g)
    # C11 = M1 - M4 + M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M3 + M6 + M7

    # We'll build the full decomposition by combining these.
    # For 2x4x5, we can view A as 2x4, B as 4x5.
    # Split A into two 2x2 blocks: A = [A_L, A_R]
    # Split B into two 2x2 blocks vertically and one 2x1 block:
    # B = [[B_LL, B_LR], [B_RL, B_RR], [B_BL, B_BR]]
    # Wait, B is 4x5. Let's split B into top 4 rows and... no, B is 4x5.
    # Let's split A into 2x2 and 2x2 horizontally.
    # Let's split B into 4x2, 4x2, 4x1 vertically? No, B is 4x5.
    # Let's split B into two 2x2 blocks and one 2x1 block vertically?
    # B = [[B11, B12, B13], [B21, B22, B23]] where each Bij is 2x?
    # Actually, B is 4x5. Let's split the 4 rows into two groups of 2.
    # So B = [B_top; B_bot] where B_top is 2x5, B_bot is 2x5.
    # Then C = A_L * B_top + A_R * B_bot.
    # Each term is 2x2 times 2x5.
    # For 2x2 times 2x5, we can split the 5 columns into two 2x2 blocks and one 2x1 block.
    # Let B_top = [B_top_L, B_top_M, B_top_R] where each is 2x2, 2x2, 2x1.
    # Then A_L * B_top = A_L * B_top_L + A_L * B_top_M + A_L * B_top_R.
    # Each 2x2 * 2x2 can use Strassen (rank 7).
    # Each 2x2 * 2x1 is just 4 terms (rank 4).
    # So for A_L * B_top: 7 + 7 + 4 = 18.
    # Similarly for A_R * B_bot: 18.
    # Total rank = 36.

    # Let's implement this.
    # We need to build U, V, W matrices.
    # U has shape (n*m, rank) = (8, rank)
    # V has shape (m*p, rank) = (20, rank)
    # W has shape (p*n, rank) = (10, rank)

    # We'll create a list of terms. Each term is a tuple (u_vec, v_vec, w_vec)
    # where u_vec is a vector of length 8, v_vec is a vector of length 20, w_vec is a vector of length 10.
    # But we can build them incrementally.

    # Let's define a helper to add a Strassen term for 2x2 * 2x2.
    # For A_block (2x2) at rows i, i+1 of A, and B_block (2x2) at cols j, j+1 of B,
    # the Strassen terms involve combinations of A and B entries.

    # Actually, it's easier to just write out the terms explicitly for the 36 rank decomposition.

    # Let's index A as A[i, j] for i in 0..1, j in 0..3.
    # Let's index B as B[j, k] for j in 0..3, k in 0..4.
    # Let's index C as C[i, k] for i in 0..1, k in 0..4.

    # U index: i*m + j = 2*i + j
    # V index: j*p + k = 5*j + k
    # W index: k*n + i = 2*k + i

    # We'll build a list of (u, v, w) where u, v, w are dicts mapping index to coefficient.

    terms = []

    # Strassen terms for 2x2 * 2x2:
    # Given A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-d)(f+h)
    # M7 = (a-c)(e+g)

    # We'll have 4 such 2x2*2x2 blocks:
    # Block 1: A_L (cols 0,1) * B_top_L (rows 0,1, cols 0,1)
    # Block 2: A_L (cols 0,1) * B_top_M (rows 0,1, cols 2,3)
    # Block 3: A_R (cols 2,3) * B_bot_L (rows 2,3, cols 0,1)
    # Block 4: A_R (cols 2,3) * B_bot_M (rows 2,3, cols 2,3)

    # And 2 2x2*2x1 blocks:
    # Block 5: A_L (cols 0,1) * B_top_R (rows 0,1, col 4)
    # Block 6: A_R (cols 2,3) * B_bot_R (rows 2,3, col 4)

    # Let's define a function to add Strassen terms for a 2x2*2x2 block.
    def add_strassen_2x2_2x2(a_rows, a_cols, b_rows, b_cols, c_cols):
        """
        a_rows: [i1, i2] rows of A
        a_cols: [j1, j2] cols of A
        b_rows: [j1, j2] rows of B
        b_cols: [k1, k2] cols of B
        c_cols: [k1, k2] cols of C
        """
        a = [(a_rows[0], a_cols[0]), (a_rows[0], a_cols[1]), (a_rows[1], a_cols[0]), (a_rows[1], a_cols[1])]
        b = [(b_rows[0], b_cols[0]), (b_rows[0], b_cols[1]), (b_rows[1], b_cols[0]), (b_rows[1], b_cols[1])]

        # M1 = (a+d)(e+h)
        # a = A[i1, j1], b = A[i1, j2], c = A[i2, j1], d = A[i2, j2]
        # e = B[j1, k1], f = B[j1, k2], g = B[j2, k1], h = B[j2, k2]
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M4
        # C21 = M2 + M5
        # C22 = M1 - M3 + M6 + M7

        # We'll add terms for each M.
        # M1: (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        # This expands to 4 terms:
        # A[i1,j1]B[j1,k1] -> C[i1,k1] += 1, C[i1,k2] += 0, C[i2,k1] += 0, C[i2,k2] += 1? No.
        # Wait, M1 contributes to C11 and C22.
        # C11 += M1, C22 += M1.
        # So the term A[i1,j1]B[j1,k1] contributes to C[i1,k1] and C[i2,k2].

        # This is getting complicated. Let's just use the rank-7 Strassen decomposition explicitly.
        # The 7 terms are:
        # 1. (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        # 2. (A21+A22)B11 -> C21 += 1
        # 3. A11(B22-B21) -> C12 += 1, C22 -= 1
        # 4. A22(B21-A21) -> C12 += 1, C21 -= 1
        # 5. (A11+A12)B22 -> C11 += 1, C21 += 1
        # 6. (A21-A22)(B22+B12) -> C22 += 1
        # 7. (A11-A21)(B11+B21) -> C11 += 1, C22 += 1

        # Let's map these to our indices.
        i1, i2 = a_rows
        j1, j2 = a_cols
        k1, k2 = b_cols

        # Term 1: (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        # Expands to 4 terms, but in the tensor decomposition, we can have a single term with coefficients.
        # U: A[i1,j1] += 1, A[i2,j2] += 1
        # V: B[j1,k1] += 1, B[j2,k2] += 1
        # W: C[i1,k1] += 1, C[i2,k2] += 1
        u = {i1*m + j1: 1, i2*m + j2: 1}
        v = {j1*p + k1: 1, j2*p + k2: 1}
        w = {k1*n + i1: 1, k2*n + i2: 1}
        terms.append((u, v, w))

        # Term 2: (A[i2,j1] + A[i2,j2]) * B[j1,k1]
        u = {i2*m + j1: 1, i2*m + j2: 1}
        v = {j1*p + k1: 1}
        w = {k1*n + i2: 1}
        terms.append((u, v, w))

        # Term 3: A[i1,j1] * (B[j2,k2] - B[j2,k1])
        u = {i1*m + j1: 1}
        v = {j2*p + k2: 1, j2*p + k1: -1}
        w = {k2*n + i1: 1, k2*n + i2: -1}  # Wait, C12 += 1, C22 -= 1
        # C12 is C[i1, k2], C22 is C[i2, k2]
        w = {k2*n + i1: 1, k2*n + i2: -1}
        terms.append((u, v, w))

        # Term 4: A[i2,j2] * (B[j2,k1] - A[i2,j1])
        # Wait, M4 = A22(B21 - A21). This is not a simple product.
        # M4 = A[i2,j2] * B[j2,k1] - A[i2,j2] * A[i2,j1]? No, M4 = A22 * (B21 - A21) is wrong.
        # M4 = A22 * (B21 - A21) is not correct.
        # Strassen M4 = A22 * (B21 - A21) is not right.
        # Correct Strassen:
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B12 - B22)
        # M4 = A22(B21 - A11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A22)(B12 + B22)
        # M7 = (A11 - A21)(B11 + B21)
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M4
        # C21 = M2 + M5
        # C22 = M1 - M3 + M6 + M7

        # So M4 = A22 * (B21 - A11). This involves A22, B21, and A11.
        # This is not a single rank-1 term. It's actually two terms: A22*B21 and -A22*A11.
        # But A22*A11 is not a matrix product term. It's a product of two entries of A.
        # This means the standard Strassen decomposition doesn't directly give rank-1 terms in the tensor sense.
        # Wait, the tensor decomposition requires each term to be a product of a vector in U, a vector in V, and a vector in W.
        # For M4 = A22 * B21 - A22 * A11, the second term A22*A11 is not of the form A[i,j]*B[k,l].
        # So I made a mistake. The Strassen decomposition for matrix multiplication tensor is not directly the algebraic Strassen algorithm.
        # The algebraic Strassen algorithm uses 7 multiplications, but each multiplication is of linear combinations.
        # In the tensor decomposition, each rank-1 term corresponds to one such multiplication.
        # So for M4 = A22 * (B21 - A11), this is not a valid term because it involves A22*A11.
        # Actually, the correct Strassen terms for the tensor are:
        # The 7 multiplications are:
        # 1. (A11+A22) * (B11+B22)
        # 2. (A21+A22) * B11
        # 3. A11 * (B12-B22)
        # 4. A22 * (B21-A11)  <-- This is wrong.
        # Let me look up the correct Strassen tensor decomposition.
        # Actually, the rank-7 decomposition of the 2x2x2 multiplication tensor is:
        # (1,1,1,1) * (1,0,0,1) * (1,0,0,1)  [M1]
        # (0,0,1,1) * (1,0,0,0) * (0,0,1,0)  [M2]
        # (1,0,0,0) * (0,1,0,-1) * (0,1,0,-1) [M3]
        # (0,0,0,1) * (0,0,1,-1) * (0,1,0,0)  [M4]
        # (1,1,0,0) * (0,0,0,1) * (1,0,1,0)  [M5]
        # (0,0,1,-1) * (0,1,0,1) * (0,0,0,1)  [M6]
        # (1,0,-1,0) * (1,0,1,0) * (1,0,0,1)  [M7]
        # Where the vectors are in the order (A11, A12, A21, A22), (B11, B12, B21, B22), (C11, C12, C21, C22).
        # Let me verify M4: (0,0,0,1) * (0,0,1,-1) * (0,1,0,0)
        # This is A22 * (B21 - B22) * C12? No, the W vector is (0,1,0,0) which is C12.
        # So M4 contributes A22*B21 to C12 and -A22*B22 to C12.
        # But the correct M4 is A22*(B21-A11). This doesn't match.
        # I think I'm confusing the algebraic algorithm with the tensor decomposition.
        # Let me just use a known rank-7 decomposition for 2x2x2.
        # One known decomposition is:
        # 1. (1,0,1,0) * (1,0,0,1) * (1,0,0,1)
        # 2. (0,0,0,1) * (1,0,0,0) * (0,0,1,0)
        # 3. (1,0,0,0) * (0,1,0,0) * (0,1,0,0)
        # 4. (0,0,0,1) * (0,0,0,1) * (0,0,0,1)
        # 5. (1,1,0,0) * (0,0,1,0) * (1,0,0,0)
        # 6. (0,0,1,1) * (0,1,0,0) * (0,0,1,0)
        # 7. (1,0,-1,0) * (1,1,0,0) * (1,1,0,0)
        # This is getting too complicated to derive by hand.

        # Let's use a simpler approach: use the rank-4 decomposition for 2x2*2x2 (which is just schoolbook) for the 2x2*2x2 blocks, and see if we can improve.
        # Schoolbook for 2x2*2x2 is rank 4.
        # So for 4 blocks: 4*4 = 16.
        # For 2 2x2*2x1 blocks: each is rank 4, so 2*4 = 8.
        # Total rank = 24.
        # This is better than 36.

        # Let's implement the schoolbook decomposition for the blocks.
        # For each 2x2*2x2 block, we have 4 terms.
        # For each 2x2*2x1 block, we have 4 terms.
        pass  # We'll implement this below.

    # Let's implement the schoolbook decomposition for the blocks.
    # Block 1: A_L (cols 0,1) * B_top_L (rows 0,1, cols 0,1)
    # A_L is 2x2, B_top_L is 2x2.
    # C = A_L * B_top_L is 2x2.
    # Schoolbook: 4 terms.
    # Term (i, j, k): A[i, j] * B[j, k] -> C[i, k]
    # For i in 0..1, j in 0..1, k in 0..1.

    # Let's create a helper to add a schoolbook term for a 2x2*2x2 block.
    def add_schoolbook_2x2_2x2(a_rows, a_cols, b_rows, b_cols, c_cols):
        i1, i2 = a_rows
        j1, j2 = a_cols
        k1, k2 = b_cols
        # j1, j2 are also the row indices of B.
        for i in [i1, i2]:
            for j in [j1, j2]:
                for k in [k1, k2]:
                    u = {i*m + j: 1}
                    v = {j*p + k: 1}
                    w = {k*n + i: 1}
                    terms.append((u, v, w))

    # Add the 4 2x2*2x2 blocks.
    # Block 1: A cols 0,1 * B rows 0,1 cols 0,1
    add_schoolbook_2x2_2x2([0, 1], [0, 1], [0, 1], [0, 1], [0, 1])
    # Block 2: A cols 0,1 * B rows 0,1 cols 2,3
    add_schoolbook_2x2_2x2([0, 1], [0, 1], [0, 1], [2, 3], [2, 3])
    # Block 3: A cols 2,3 * B rows 2,3 cols 0,1
    add_schoolbook_2x2_2x2([0, 1], [2, 3], [2, 3], [0, 1], [0, 1])
    # Block 4: A cols 2,3 * B rows 2,3 cols 2,3
    add_schoolbook_2x2_2x2([0, 1], [2, 3], [2, 3], [2, 3], [2, 3])

    # Add the 2 2x2*2x1 blocks.
    # Block 5: A cols 0,1 * B rows 0,1 col 4
    def add_schoolbook_2x2_2x1(a_rows, a_cols, b_rows, b_col, c_col):
        i1, i2 = a_rows
        j1, j2 = a_cols
        k = b_col
        for i in [i1, i2]:
            for j in [j1, j2]:
                u = {i*m + j: 1}
                v = {j*p + k: 1}
                w = {k*n + i: 1}
                terms.append((u, v, w))

    add_schoolbook_2x2_2x1([0, 1], [0, 1], [0, 1], 4, 4)
    add_schoolbook_2x2_2x1([0, 1], [2, 3], [2, 3], 4, 4)

    # Now, we have a list of terms. Each term is (u, v, w) where u, v, w are dicts.
    # We need to merge them into U, V, W matrices.
    # The total number of terms is 4*4 + 2*4 = 16 + 8 = 24.
    # So rank = 24.

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for col, (u, v, w) in enumerate(terms):
        for idx, coeff in u.items():
            U[idx][col] += coeff
        for idx, coeff in v.items():
            V[idx][col] += coeff
        for idx, coeff in w.items():
            W[idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
