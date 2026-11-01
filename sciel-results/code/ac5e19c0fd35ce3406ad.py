DIMENSIONS = [2, 4, 6]
import time
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-42 exact tensor decomposition for 2x4 by 4x6 matrix multiplication,
    constructed as the Kronecker product of a 2x2x2 Strassen decomposition (rank 7)
    and a 1x2x3 schoolbook decomposition (rank 6).
    Rank = 7 * 6 = 42.
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=6

    # Strassen 2x2x2 decomposition (rank 7)
    # U1: 4 x 7, V1: 4 x 7, W1: 4 x 7
    U1 = [
        [1,  0,  1,  0,  1, -1,  0],  # a
        [0,  0,  0,  0,  1,  0,  1],  # b
        [0,  1,  0,  0,  0,  1,  0],  # c
        [1,  1,  0,  1,  0,  0, -1]   # d
    ]
    V1 = [
        [1,  1,  0, -1,  0,  1,  0],  # x
        [0,  0,  1,  0,  0,  1,  0],  # y
        [0,  0,  0,  1,  0,  0,  1],  # z
        [1,  0, -1,  0,  1,  0,  1]   # w
    ]
    W1 = [
        [1,  0,  0,  1, -1,  0,  1],  # k=0, i=0
        [0,  0,  1,  0,  1,  0,  0],  # k=0, i=1
        [0,  1,  0,  1,  0,  0,  0],  # k=1, i=0
        [1, -1,  1,  0,  0,  1,  0]   # k=1, i=1
    ]

    # Schoolbook 1x2x3 decomposition (rank 6)
    # U2: 2 x 6, V2: 6 x 6, W2: 3 x 6
    U2 = [
        [1, 0, 1, 0, 1, 0],  # a0
        [0, 1, 0, 1, 0, 1]   # a1
    ]
    V2 = [
        [1, 0, 0, 0, 0, 0],  # b00
        [0, 1, 0, 0, 0, 0],  # b01
        [0, 0, 1, 0, 0, 0],  # b02
        [0, 0, 0, 1, 0, 0],  # b10
        [0, 0, 0, 0, 1, 0],  # b11
        [0, 0, 0, 0, 0, 1]   # b12
    ]
    W2 = [
        [1, 0, 0, 1, 0, 0],  # k=0, i=0
        [0, 1, 0, 0, 1, 0],  # k=1, i=0
        [0, 0, 1, 0, 0, 1]   # k=2, i=0
    ]

    rank = 7 * 6  # 42
    U = [[0] * rank for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank for _ in range(p * n)]   # 12 x 42

    # The Kronecker product structure:
    # We view the 2x4x6 multiplication as (2x2 by 2x2) for the "Strassen part"
    # and (1x2 by 2x3) for the "schoolbook part".
    #
    # The 2x4 matrix A is split into 2 row-blocks of size 1x4, each further split
    # into 2 sub-blocks of size 1x2. So A[i][j] where i in {0,1}, j in {0,1,2,3}.
    # Let i = 2*i1 + i0 won't work directly since n=2, m=4.
    #
    # Better: Think of A as a 2x4 matrix. Split rows into 2 groups of 1? No.
    # Split the 2 rows as the "2" in Strassen, and the 4 columns as 2 groups of 2
    # for the "2" in the 1x2 part.
    #
    # A[i][j], i in {0,1}, j in {0,1,2,3}
    # B[j][k], j in {0,1,2,3}, k in {0,1,2,3,4,5}
    #
    # For Strassen on the 2x2 "row" aspect:
    #   We treat the 2 rows as the first dimension of Strassen.
    #   The 4 columns are split into 2 blocks of 2: j1 in {0,1} where j = 2*j1 + j0.
    #   For each Strassen factor r1 in {0..6}, we need a 1x2x3 multiplication
    #   for each column-block.
    #
    # Actually, let's use the proper Kronecker product indexing:
    # U_total[i, j; r1, r2] = U1[i, r1] * U2[j, r2]
    # where i indexes the 2x2 Strassen part, j indexes the 1x2 schoolbook part.
    # The combined input index is (i, j) -> flattened.
    #
    # For A (2x4): index (row, col) where row in {0,1}, col in {0,1,2,3}
    #   Strassen row: i = row (0 or 1)
    #   Schoolbook col: j = col // 2 (0 or 1), sub-col = col % 2
    #   But U2 has 2 rows (a0, a1) corresponding to the 2 columns in the 1x2 part.
    #   So U2 row index = col % 2? No, U2 is 1x2 meaning 1 row, 2 cols.
    #   In the tensor, U has n*m = 8 rows. Let's map:
    #   A_flat[row * 4 + col]
    #   Strassen index: s_row = row
    #   Schoolbook index: s_col = col // 2 (which block of 2), s_subcol = col % 2
    #   U2 is for a 1x2 input, so its row index corresponds to s_subcol? 
    #   Actually U2 has shape (1*2, 6) = (2, 6). Row 0 = a0, Row 1 = a1.
    #   These correspond to the 2 elements in the 1x2 matrix.
    #   In our 4-column split into 2 blocks of 2, for block s_col, the elements are
    #   A[row][2*s_col + 0] and A[row][2*s_col + 1].
    #   So U2 row index = 2*s_col + s_subcol? No, that's 4 values but U2 only has 2 rows.
    #
    # Let me reconsider. The Kronecker product of 2x2x2 and 1x2x3 gives 2*1 x 2*2 x 2*3 = 2x4x6.
    # Input 1 (2x2) from Strassen, Input 2 (1x2) from schoolbook.
    # The combined 2x4 matrix: rows from 2x2 (2 rows), columns from 2*2 (4 cols).
    # U_combined has 2*4 = 8 rows.
    # Row index = (r_strassen, r_schoolbook) where r_strassen in {0,1}, r_schoolbook in {0,1,2,3}.
    # But schoolbook is 1x2, so it has 2 elements. The 4 columns come from 2 (from 2x2) * 2 (from 1x2).
    # So column index c = 2*c_strassen + c_schoolbook where c_strassen in {0,1}, c_schoolbook in {0,1}.
    # U_combined row index (flattened) = r_strassen * 4 + (2*c_strassen + c_schoolbook)
    # U_combined[r_strassen, c; r1, r2] = U1[r_strassen, r1] * U2[c_schoolbook, r2] * delta(c_strassen, ...)
    # Hmm, this is getting complicated. Let me just do it directly.
    #
    # Simpler approach: The Kronecker product of tensors T1 (2x2x2) and T2 (1x2x3)
    # gives T (2*1 x 2*2 x 2*3) = (2x4x6).
    # U_kronecker[i, j; a, b] = U1[i, a] * U2[j, b] where i in 0..1, j in 0..3 (for 2*2=4)
    # Wait, U2 is 1x2 so it has 1*2=2 rows. But we need 4 columns for the 2x4 matrix.
    # The 4 columns come from the 2 columns of the 2x2 (Strassen) times the 2 columns of the 1x2 (schoolbook).
    # So the column index in the 2x4 matrix is (c1, c2) where c1 in {0,1} (from 2x2), c2 in {0,1} (from 1x2).
    # Flat column index = c1 * 2 + c2.
    # U1 row index = i (0 or 1) for the 2 rows.
    # U2 row index = c2 (0 or 1) for the 2 elements in the 1x2.
    # But we also need to account for c1. The Kronecker product structure means:
    # U_combined[i, c1*2+c2; a, b] = U1[i, a] * U2[c2, b] * (some factor for c1)
    # Actually no. In a Kronecker product of matrix multiplications, the structure is:
    # (A1 ⊗ A2) * (B1 ⊗ B2) = (A1*B1) ⊗ (A2*B2)
    # where A1 is 2x2, A2 is 1x2, B1 is 2x2, B2 is 2x3.
    # So A = A1 ⊗ A2 is 2*1 x 2*2 = 2x4. ✓
    # B = B1 ⊗ B2 is 2*2 x 2*3 = 4x6. ✓
    # C = C1 ⊗ C2 is 2*1 x 2*3 = 2x6? No, C should be 2x6.
    # C1 is 2x2 (output of 2x2*2x2), C2 is 1x3 (output of 1x2*2x3).
    # C = C1 ⊗ C2 is 2*1 x 2*3 = 2x6. ✓
    #
    # Now the tensor decomposition:
    # U_combined for A = A1 ⊗ A2:
    # A[row, col] where row in {0,1}, col in {0,1,2,3}
    # col = 2*c1 + c2 where c1 in {0,1}, c2 in {0,1}
    # A[row, col] = A1[row, c1] * A2[0, c2] (since A2 is 1x2)
    # U_combined[row, col; r1, r2] = U1[row, r1] * U2[c2, r2]
    # But this doesn't depend on c1! That can't be right.
    #
    # I think the issue is that A2 is 1x2, so it only has one row. The "2" in the 2x4 columns
    # from A1's 2 columns and A2's 2 columns. But A1 is 2x2, so A1[row, c1] for c1 in {0,1}.
    # And A2 is 1x2, so A2[0, c2] for c2 in {0,1}.
    # A[row, 2*c1+c2] = A1[row, c1] * A2[0, c2]
    # So U_combined[row, 2*c1+c2; r1, r2] = U1[row, r1] * U2[c2, r2]
    # This means for fixed row and c1, the values for c2=0 and c2=1 differ only by U2[c2, r2].
    # And c1 doesn't appear! That's a problem.
    #
    # I see the issue: the Kronecker product structure requires that the decomposition
    # respects the tensor product structure. For the U matrix, the index (row, col) maps to
    # (row_A1, col_A1, col_A2) but since A2 has only 1 row, the row index of A is just row_A1.
    # The col index of A is (col_A1, col_A2).
    # U_A[(row_A1, col_A1, col_A2), (r1, r2)] = U_A1[row_A1, r1] * U_A2[col_A2, r2]
    # where U_A2 is the U for the 1x2 part, which has 1*2=2 rows indexed by col_A2.
    # So U_A2[col_A2, r2] where col_A2 in {0,1}.
    # And col_A1 doesn't appear in U_A! It's only in the V and W matrices.
    #
    # Let's verify with V:
    # B is 4x6. B = B1 ⊗ B2 where B1 is 2x2, B2 is 2x3.
    # B[row, col] where row in {0,1,2,3}, col in {0,1,2,3,4,5}
    # row = 2*row_B1 + row_B2 where row_B1 in {0,1}, row_B2 in {0,1}
    # col = 3*col_B1 + col_B2 where col_B1 in {0,1}, col_B2 in {0,1,2}
    # B[row, col] = B1[row_B1, col_B1] * B2[row_B2, col_B2]
    # V_B[row, col; s1, s2] = V_B1[row_B1, s1] * V_B2[col_B1, s2] * (something for row_B2, col_B2)
    # Actually: V_B[row, col; s1, s2] = V_B1[row_B1, s1] * V_B2[row_B2, col_B2, s2]? No.
    # V for B2 (2x3) has 2*3=6 rows indexed by (row_B2, col_B2).
    # V_B2[(row_B2, col_B2), s2]
    # V_B1[row_B1, s1] where row_B1 in {0,1}.
    # V_B[(2*row_B1+row_B2, 3*col_B1+col_B2), (s1, s2)] = V_B1[row_B1, s1] * V_B2[(row_B2, col_B2), s2]
    #
    # And for W:
    # C is 2x6. C = C1 ⊗ C2 where C1 is 2x2, C2 is 1x3.
    # C[row, col] where row in {0,1}, col in {0,1,2,3,4,5}
    # row = row_C1 (since C2 has 1 row)
    # col = 3*col_C1 + col_C2 where col_C1 in {0,1}, col_C2 in {0,1,2}
    # C[row, col] = C1[row, col_C1] * C2[0, col_C2]
    # W_C[row, col; t1, t2] = W_C1[row, t1] * W_C2[col_C2, t2]
    # where W_C2 is for the 1x3 part, with 1*3=3 rows indexed by col_C2.
    #
    # Now let's code this up properly.

    # U1: 2x2 part, 4 rows (2*2), 7 cols
    # Wait, U1 for 2x2x2 should have n*m = 2*2 = 4 rows. But above I defined U1 as 4x7.
    # The 4 rows correspond to (i, j) where i in {0,1} (row), j in {0,1} (col) of the 2x2.
    # So U1[i*2+j, r1]
    # In my U1 above, row 0 = a (i=0,j=0), row 1 = b (i=0,j=1), row 2 = c (i=1,j=0), row 3 = d (i=1,j=1).
    # Yes, that's i*2+j.

    # For the combined 2x4 matrix A:
    # A[row, col] where row in {0,1}, col in {0,1,2,3}
    # row = row_A1 (just row, since A2 has 1 row)
    # col = 2*col_A1 + col_A2 where col_A1 in {0,1}, col_A2 in {0,1}
    # U_A[row, col; r1, r2] = U1[row*2+col_A1, r1] * U2[col_A2, r2]
    # U_A flat row index = row*4 + col

    # V for B (4x6):
    # B[row, col] where row in {0..3}, col in {0..5}
    # row = 2*row_B1 + row_B2, col = 3*col_B1 + col_B2
    # V_B[row, col; s1, s2] = V_B1[row_B1*2+col_B1, s1] * V_B2[(row_B2, col_B2), s2]
    # V_B1 is 4x7 (for 2x2), indexed by (row_B1, col_B1) -> row_B1*2+col_B1
    # V_B2 is 6x6 (for 2x3), indexed by (row_B2, col_B2) -> row_B2*3+col_B2

    # W for C (2x6):
    # C[row, col] where row in {0,1}, col in {0..5}
    # row = row_C1, col = 3*col_C1 + col_C2
    # W_C[row, col; t1, t2] = W_C1[row*2+col_C1, t1] * W_C2[col_C2, t2]
    # W_C1 is 4x7 (for 2x2), indexed by (row, col_C1) -> row*2+col_C1
    # W_C2 is 3x6 (for 1x3), indexed by col_C2

    # Now fill in the matrices:
    for r1 in range(7):
        for r2 in range(6):
            r = r1 * 6 + r2  # combined rank index
            # U
            for row in range(2):
                for col_A1 in range(2):
                    for col_A2 in range(2):
                        col = col_A1 * 2 + col_A2
                        u_row = row * 4 + col
                        u1_val = U1[row * 2 + col_A1][r1]
                        u2_val = U2[col_A2][r2]
                        U[u_row][r] += u1_val * u2_val

            # V
            for row_B1 in range(2):
                for col_B1 in range(2):
                    for row_B2 in range(2):
                        for col_B2 in range(3):
                            v_row = row_B1 * 2 + row_B2  # wait, this is wrong
                            # B is 4x6. Row index = row_B1 * 2 + row_B2? No.
                            # B row = 2*row_B1 + row_B2 where row_B1 in {0,1}, row_B2 in {0,1}.
                            # So row = row_B1 * 2 + row_B2.
                            # B col = 3*col_B1 + col_B2 where col_B1 in {0,1}, col_B2 in {0,1,2}.
                            # So col = col_B1 * 3 + col_B2.
                            v_row = row_B1 * 2 + row_B2
                            v_col = col_B1 * 3 + col_B2
                            v1_val = V1[row_B1 * 2 + col_B1][r1]
                            v2_val = V2[row_B2 * 3 + col_B2][r2]
                            V[v_row * 6 + v_col][r] += v1_val * v2_val

            # W
            for row_C1 in range(2):
                for col_C1 in range(2):
                    for col_C2 in range(3):
                        w_row = row_C1  # C is 2x6, but W is p*n = 6*2 = 12 rows
                        # W index is k*n + i where k is col (0..5), i is row (0..1)
                        # k = 3*col_C1 + col_C2, i = row_C1
                        w_flat_row = (3 * col_C1 + col_C2) * 2 + row_C1
                        w1_val = W1[row_C1 * 2 + col_C1][r1]
                        w2_val = W2[col_C2][r2]
                        W[w_flat_row][r] += w1_val * w2_val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
