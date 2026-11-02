"""Matrix multiplication 2x4x7 with rank-minimal decomposition using Kronecker/block structure."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    # Dimensions
    n, m, p = 2, 4, 7
    # rank = 50
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]  # 8 x 50
    V = [[0.0] * rank for _ in range(m * p)]  # 28 x 50
    W = [[0.0] * rank for _ in range(p * n)]  # 14 x 50

    # Strassen 7-term decomposition for 2x2x2 matrices.
    # For A (2x2), B (2x2), C = A*B (2x2):
    # C[0,0] = M1 - M2 + M4 + M6  -> but let's use the standard 7 multiplications
    # Standard Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # We have two independent blocks of A (rows 0-1 and 2-3), each 2x4.
    # We split B into 4 column blocks: B[:,0:2], B[:,2:4], B[:,4:6], B[:,6:7]
    # The first three are 4x2, the last is 4x1.

    col = 0
    for a_block in range(2):
        # A_block is rows [2*a_block : 2*a_block+2], i.e., 2x4
        # B_block is rows [2*a_block : 2*a_block+2], i.e., 4x7
        # We need to compute C_block = A_block * B_block, which is 2x7

        for k_block in range(4):
            if k_block < 3:
                # 2x2x2 Strassen decomposition
                # A_sub: 2x2 (columns 2*k_block : 2*k_block+2 of A_block)
                # B_sub: 2x2 (rows 2*a_block : 2*a_block+2, cols 2*k_block : 2*k_block+2 of B)
                # C_sub: 2x2 (rows 0:2, cols 2*k_block : 2*k_block+2 of C_block)

                # A_sub indices in U:
                # A_sub[0,0] -> U row (2*a_block)*4 + 2*k_block
                # A_sub[0,1] -> U row (2*a_block)*4 + 2*k_block + 1
                # A_sub[1,0] -> U row (2*a_block+1)*4 + 2*k_block
                # A_sub[1,1] -> U row (2*a_block+1)*4 + 2*k_block + 1

                # B_sub indices in V:
                # B_sub[0,0] -> V row (2*a_block)*7 + 2*k_block
                # B_sub[0,1] -> V row (2*a_block)*7 + 2*k_block + 1
                # B_sub[1,0] -> V row (2*a_block+1)*7 + 2*k_block
                # B_sub[1,1] -> V row (2*a_block+1)*7 + 2*k_block + 1

                # C_sub indices in W:
                # C_sub[0,0] -> W row (2*k_block)*2 + 0
                # C_sub[0,1] -> W row (2*k_block)*2 + 1
                # C_sub[1,0] -> W row (2*k_block+1)*2 + 0
                # C_sub[1,1] -> W row (2*k_block+1)*2 + 1

                A00 = (2*a_block) * 4 + 2*k_block
                A01 = (2*a_block) * 4 + 2*k_block + 1
                A10 = (2*a_block + 1) * 4 + 2*k_block
                A11 = (2*a_block + 1) * 4 + 2*k_block + 1

                B00 = (2*a_block) * 7 + 2*k_block
                B01 = (2*a_block) * 7 + 2*k_block + 1
                B10 = (2*a_block + 1) * 7 + 2*k_block
                B11 = (2*a_block + 1) * 7 + 2*k_block + 1

                C00 = (2*k_block) * 2 + 0
                C01 = (2*k_block) * 2 + 1
                C10 = (2*k_block + 1) * 2 + 0
                C11 = (2*k_block + 1) * 2 + 1

                # M1 = (A00+A11)*(B00+B11)
                U[A00][col] += 1
                U[A11][col] += 1
                V[B00][col] += 1
                V[B11][col] += 1
                W[C00][col] += 1
                W[C01][col] += 1
                W[C10][col] += 1
                W[C11][col] -= 1
                col += 1

                # M2 = (A10+A11)*B00
                U[A10][col] += 1
                U[A11][col] += 1
                V[B00][col] += 1
                W[C10][col] += 1
                W[C11][col] -= 1
                col += 1

                # M3 = A00*(B01-B11)
                U[A00][col] += 1
                V[B01][col] += 1
                V[B11][col] -= 1
                W[C01][col] += 1
                W[C11][col] += 1
                col += 1

                # M4 = A11*(B10-B00)
                U[A11][col] += 1
                V[B10][col] += 1
                V[B00][col] -= 1
                W[C00][col] += 1
                W[C10][col] += 1
                col += 1

                # M5 = (A00+A01)*B11
                U[A00][col] += 1
                U[A01][col] += 1
                V[B11][col] += 1
                W[C00][col] -= 1
                W[C01][col] += 1
                col += 1

                # M6 = (A10-A00)*(B00+B01)
                U[A10][col] += 1
                U[A00][col] -= 1
                V[B00][col] += 1
                V[B01][col] += 1
                W[C11][col] += 1
                col += 1

                # M7 = (A01-A11)*(B10+B11)
                U[A01][col] += 1
                U[A11][col] -= 1
                V[B10][col] += 1
                V[B11][col] += 1
                W[C00][col] += 1
                col += 1
            else:
                # k_block == 3: 2x2x1 schoolbook
                # A_sub: 2x2 (columns 6:7... wait, 2*3=6, so cols 6:8? But p=7, so only col 6)
                # Actually, for k_block=3, we have columns 2*3=6 to 2*4=8, but B only has 7 columns (0-6)
                # So we only have B[:,6], which is 4x1.
                # A_block is 2x4, so A_block[:,6] doesn't exist. A_block only has 4 columns (0-3).
                # Wait, I made a mistake. Let me re-examine the dimensions.

                # A is 2x4, B is 4x7. So A has 4 columns, B has 7 columns.
                # When I split A into 2 blocks of 2 rows, each A_block is 2x4.
                # When I split B into column blocks, I should split B's 7 columns.
                # But Strassen requires square matrices. So I should split A's 4 columns into 2 blocks of 2,
                # and B's 7 columns into... this is getting complicated.

                # Let me reconsider the approach. Maybe I should use a different decomposition.

                # Actually, let me just use the schoolbook for this part.
                # A_block is 2x4, B_block is 4x1 (column 6 of B).
                # C_block is 2x1.
                # C[0,6] = A[0,0]*B[0,6] + A[0,1]*B[1,6] + A[0,2]*B[2,6] + A[0,3]*B[3,6]
                # C[1,6] = A[1,0]*B[0,6] + A[1,1]*B[1,6] + A[1,2]*B[2,6] + A[1,3]*B[3,6]

                # This is 8 rank-1 terms, not 4. So my earlier calculation was wrong.

                # Let me recalculate the total rank:
                # 2 A-blocks * (3 Strassen blocks * 7 terms + 1 schoolbook block * 8 terms) = 2 * (21 + 8) = 2 * 29 = 58
                # That's worse than 56!

                # I need to find a better approach. Let me just return the schoolbook for now.
                # Actually, let me try a different decomposition.

                # Let me use the fact that 4 = 2*2 and 7 = 4+3.
                # Or maybe I should just use the schoolbook decomposition, which gives rank 56.
                # The current score is 1.0000, which corresponds to rank 56.
                # I need to find a rank < 56.

                # Let me try a simpler approach: use 2x2 Strassen for the 2x4x4 part, and handle the 2x4x3 part separately.
                # 2x4x4: split A into 2 blocks of 2x2, B into 2 blocks of 2x4, C into 2 blocks of 2x4.
                # Each 2x2x4 can be done with 2 Strassen 2x2x2 blocks: 2*7 = 14 terms.
                # Total for 2x4x4: 2 * 14 = 28 terms.
                # 2x4x3: schoolbook: 2*4*3 = 24 terms.
                # Total: 28 + 24 = 52 terms. That's better than 56!

                # Let me implement this.
                pass  # We'll handle this in a different way below

    # The above approach is getting too complicated. Let me just implement the 52-rank decomposition directly.
    # Reset U, V, W
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0

    # Part 1: 2x4x4 using Strassen
    # A is 2x4, B is 4x4 (cols 0-3), C is 2x4 (cols 0-3)
    # Split A into A0 (rows 0-1, cols 0-1) and A1 (rows 0-1, cols 2-3), each 2x2
    # Split B into B0 (rows 0-3, cols 0-1) and B1 (rows 0-3, cols 2-3), each 4x2
    # Wait, Strassen requires both matrices to be 2x2. So I need to split B into 2x2 blocks too.
    # B is 4x4, so I can split it into 4 2x2 blocks: B[0:2,0:2], B[0:2,2:4], B[2:4,0:2], B[2:4,2:4]
    # A is 2x4, so I can split it into 2 2x2 blocks: A[0:2,0:2], A[0:2,2:4]
    # C is 2x4, so I can split it into 2 2x2 blocks: C[0:2,0:2], C[0:2,2:4]

    # C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2:4]*B[2:4,0:2]
    # C[0:2,2:4] = A[0:2,0:2]*B[0:2,2:4] + A[0:2,2:4]*B[2:4,2:4]

    # Each product is a sum of two 2x2*2x2 products, each using 7 Strassen terms.
    # So total: 4 * 7 = 28 terms for the 2x4x4 part.

    # Part 2: 2x4x3 using schoolbook
    # A is 2x4, B is 4x3 (cols 4-6), C is 2x3 (cols 4-6)
    # Schoolbook: 2*4*3 = 24 terms.

    # Total: 28 + 24 = 52 terms.

    # Let me implement this.

    # Part 1: 2x4x4
    # A0 = A[0:2, 0:2] (rows 0-1, cols 0-1)
    # A1 = A[0:2, 2:4] (rows 0-1, cols 2-3)
    # B00 = B[0:2, 0:2]
    # B01 = B[0:2, 2:4]
    # B10 = B[2:4, 0:2]
    # B11 = B[2:4, 2:4]
    # C0 = C[0:2, 0:2]
    # C1 = C[0:2, 2:4]

    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11

    def strassen_term(A_row0, A_row1, A_col0, A_col1, B_row0, B_row1, B_col0, B_col1,
                      C_row0, C_row1, C_col0, C_col1, term, U, V, W, col):
        """Add one Strassen term to U, V, W."""
        A00 = A_row0 * 4 + A_col0
        A01 = A_row0 * 4 + A_col1
        A10 = A_row1 * 4 + A_col0
        A11 = A_row1 * 4 + A_col1

        B00 = B_row0 * 7 + B_col0
        B01 = B_row0 * 7 + B_col1
        B10 = B_row1 * 7 + B_col0
        B11 = B_row1 * 7 + B_col1

        C00 = C_row0 * 2 + C_col0
        C01 = C_row0 * 2 + C_col1
        C10 = C_row1 * 2 + C_col0
        C11 = C_row1 * 2 + C_col1

        if term == 0:  # M1 = (A00+A11)*(B00+B11)
            U[A00][col] += 1
            U[A11][col] += 1
            V[B00][col] += 1
            V[B11][col] += 1
            W[C00][col] += 1
            W[C01][col] += 1
            W[C10][col] += 1
            W[C11][col] -= 1
        elif term == 1:  # M2 = (A10+A11)*B00
            U[A10][col] += 1
            U[A11][col] += 1
            V[B00][col] += 1
            W[C10][col] += 1
            W[C11][col] -= 1
        elif term == 2:  # M3 = A00*(B01-B11)
            U[A00][col] += 1
            V[B01][col] += 1
            V[B11][col] -= 1
            W[C01][col] += 1
            W[C11][col] += 1
        elif term == 3:  # M4 = A11*(B10-B00)
            U[A11][col] += 1
            V[B10][col] += 1
            V[B00][col] -= 1
            W[C00][col] += 1
            W[C10][col] += 1
        elif term == 4:  # M5 = (A00+A01)*B11
            U[A00][col] += 1
            U[A01][col] += 1
            V[B11][col] += 1
            W[C00][col] -= 1
            W[C01][col] += 1
        elif term == 5:  # M6 = (A10-A00)*(B00+B01)
            U[A10][col] += 1
            U[A00][col] -= 1
            V[B00][col] += 1
            V[B01][col] += 1
            W[C11][col] += 1
        elif term == 6:  # M7 = (A01-A11)*(B10+B11)
            U[A01][col] += 1
            U[A11][col] -= 1
            V[B10][col] += 1
            V[B11][col] += 1
            W[C00][col] += 1

    # C0 = A0*B00 + A1*B10
    for t in range(7):
        # A0*B00
        strassen_term(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, t, U, V, W, col)
        col += 1
    for t in range(7):
        # A1*B10
        strassen_term(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, t, U, V, W, col)
        col += 1

    # C1 = A0*B01 + A1*B11
    for t in range(7):
        # A0*B01
        strassen_term(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, t, U, V, W, col)
        col += 1
    for t in range(7):
        # A1*B11
        strassen_term(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3, t, U, V, W, col)
        col += 1

    # Part 2: 2x4x3 schoolbook
    # A is 2x4, B is 4x3 (cols 4-6), C is 2x3 (cols 4-6)
    # C[i, 4+j] = sum_k A[i, k] * B[k, 4+j]
    for i in range(2):
        for k in range(4):
            for j in range(3):
                U[i * 4 + k][col] += 1
                V[k * 7 + 4 + j][col] += 1
                W[(4 + j) * 2 + i][col] += 1
                col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
