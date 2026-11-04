"""Strassen-based rank-28 decomposition for 2x4 by 4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen 2x2 multiplication products
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

    # We split the 4 columns of A into two blocks of 2:
    # Block 1: columns 0,1 -> A1 (2x2)
    # Block 2: columns 2,3 -> A2 (2x2)
    #
    # We split the 4 rows of B into two blocks of 2:
    # Top: rows 0,1 -> B_top (2x4)
    # Bottom: rows 2,3 -> B_bottom (2x4)
    #
    # For each (A_block, B_block) pair, we do a Strassen 2x2 multiplication
    # to get a 2x2 result block, which contributes to the final 2x4 output.
    #
    # Actually, let's think more carefully.
    # A is 2x4, B is 4x4, C is 2x4.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Split A into A_left (2x2, cols 0,1) and A_right (2x2, cols 2,3)
    # Split B into B_top (2x4, rows 0,1) and B_bottom (2x4, rows 2,3)
    #
    # C = A_left * B_top + A_right * B_bottom
    #
    # Each term A_left * B_top is 2x2 * 2x4 = 2x4.
    # We can compute A_left * B_top by splitting B_top into B_top_left (2x2) and B_top_right (2x2).
    # Then A_left * B_top_left is 2x2*2x2, computed via Strassen (rank 7).
    # And A_left * B_top_right is 2x2*2x2, computed via Strassen (rank 7).
    # Similarly for A_right * B_bottom.
    # Total rank = 7+7+7+7 = 28.

    def add_strassen_block(a_row0, a_row1, a_col0, a_col1, b_row0, b_row1, b_col0, b_col1, 
                           c_row0, c_row1, c_col0, c_col1, offset):
        """Add a Strassen 2x2x2 block at given global indices, starting at rank offset."""
        # A indices: rows a_row0,a_row1; cols a_col0,a_col1
        # B indices: rows b_row0,b_row1; cols b_col0,b_col1
        # C indices: rows c_row0,c_row1; cols c_col0,c_col1
        # 
        # Strassen products M1..M7
        # M1 = (A11+A22)(B11+B22)
        U[a_row0*m + a_col0][offset + 0] += 1
        U[a_row1*m + a_col1][offset + 0] += 1
        V[b_row0*p + b_col0][offset + 0] += 1
        V[b_row1*p + b_col1][offset + 0] += 1
        W[c_col0*n + c_row0][offset + 0] += 1   # C11 gets M1
        W[c_col0*n + c_row1][offset + 0] += 1   # C21 gets M1? No...
        
        # Wait, I need to be careful with the C index mapping.
        # C is 2x4. C[i,k] where i in {0,1}, k in {0,1,2,3}.
        # W index is k*n + i = 4*k + i.
        # The 2x2 result block has rows c_row0,c_row1 and cols c_col0,c_col1.
        # So C[c_row0, c_col0] gets contributions, C[c_row0, c_col1], C[c_row1, c_col0], C[c_row1, c_col1].
        # W indices: c_col0*n + c_row0, c_col1*n + c_row0, c_col0*n + c_row1, c_col1*n + c_row1.

        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        W[c_col0*n + c_row0][offset + 0] += 0  # already added
        W[c_col1*n + c_row1][offset + 0] += 1  # C22 gets M1

        # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
        U[a_row1*m + a_col0][offset + 1] += 1
        U[a_row1*m + a_col1][offset + 1] += 1
        V[b_row0*p + b_col0][offset + 1] += 1
        W[c_col0*n + c_row1][offset + 1] += 1  # C21 gets M2
        W[c_col1*n + c_row1][offset + 1] -= 1  # C22 gets -M2

        # M3 = A11(B12-B22) -> C12 += M3, C22 += M3
        U[a_row0*m + a_col0][offset + 2] += 1
        V[b_row0*p + b_col1][offset + 2] += 1
        V[b_row1*p + b_col1][offset + 2] -= 1
        W[c_col1*n + c_row0][offset + 2] += 1  # C12 gets M3
        W[c_col1*n + c_row1][offset + 2] += 1  # C22 gets M3

        # M4 = A22(B21-B11) -> C11 += M4, C21 += M4
        U[a_row1*m + a_col1][offset + 3] += 1
        V[b_row1*p + b_col0][offset + 3] += 1
        V[b_row0*p + b_col0][offset + 3] -= 1
        W[c_col0*n + c_row0][offset + 3] += 1  # C11 gets M4
        W[c_col0*n + c_row1][offset + 3] += 1  # C21 gets M4

        # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
        U[a_row0*m + a_col0][offset + 4] += 1
        U[a_row0*m + a_col1][offset + 4] += 1
        V[b_row1*p + b_col1][offset + 4] += 1
        W[c_col0*n + c_row0][offset + 4] -= 1  # C11 gets -M5
        W[c_col1*n + c_row0][offset + 4] += 1  # C12 gets M5

        # M6 = (A21-A11)(B11+B12) -> C22 += M6
        U[a_row1*m + a_col0][offset + 5] += 1
        U[a_row0*m + a_col0][offset + 5] -= 1
        V[b_row0*p + b_col0][offset + 5] += 1
        V[b_row0*p + b_col1][offset + 5] += 1
        W[c_col1*n + c_row1][offset + 5] += 1  # C22 gets M6

        # M7 = (A12-A22)(B21+B22) -> C11 += M7
        U[a_row0*m + a_col1][offset + 6] += 1
        U[a_row1*m + a_col1][offset + 6] -= 1
        V[b_row1*p + b_col0][offset + 6] += 1
        V[b_row1*p + b_col1][offset + 6] += 1
        W[c_col0*n + c_row0][offset + 6] += 1  # C11 gets M7

    # Block 1: A_left (rows 0,1; cols 0,1) * B_top_left (rows 0,1; cols 0,1) -> C block (rows 0,1; cols 0,1)
    add_strassen_block(0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0)

    # Block 2: A_left (rows 0,1; cols 0,1) * B_top_right (rows 0,1; cols 2,3) -> C block (rows 0,1; cols 2,3)
    add_strassen_block(0, 1, 0, 1, 0, 1, 2, 3, 0, 1, 2, 3, 7)

    # Block 3: A_right (rows 0,1; cols 2,3) * B_bottom_left (rows 2,3; cols 0,1) -> C block (rows 0,1; cols 0,1)
    add_strassen_block(0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 0, 1, 14)

    # Block 4: A_right (rows 0,1; cols 2,3) * B_bottom_right (rows 2,3; cols 2,3) -> C block (rows 0,1; cols 2,3)
    add_strassen_block(0, 1, 2, 3, 2, 3, 2, 3, 0, 1, 2, 3, 21)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
