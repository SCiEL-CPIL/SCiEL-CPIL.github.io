def construct():
    n, m, p = 3, 4, 4
    rank = 36
    U = [[0]*rank for _ in range(n*m)]  # 12 x 36
    V = [[0]*rank for _ in range(m*p)]  # 16 x 36
    W = [[0]*rank for _ in range(p*n)]  # 12 x 36

    col = 0
    # We split B into left (cols 0,1) and right (cols 2,3)
    # For each side, we compute 3x2 result
    # Split A into top (rows 0,1) and bottom (row 2)
    # Top part: 2x4 * 4x2
    # Split A_top into A_TL (2x2, cols 0,1) and A_TR (2x2, cols 2,3)
    # B_left is 4x2: rows 0,1 form B_TL (2x2), rows 2,3 form B_BL (2x2)
    # A_top * B_left = A_TL * B_TL + A_TR * B_BL
    # Use Strassen for each 2x2*2x2 -> rank 7 each, total 14
    # Bottom part: 1x4 * 4x2 -> rank 4
    # Total per side: 18
    # Two sides: 36

    def strassen_block(U_mat, V_mat, W_mat, a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, col_base):
        # Standard Strassen for 2x2 * 2x2
        # a = A[a_row_off:a_row_off+2, a_col_off:a_col_off+2]
        # b = B[b_row_off:b_row_off+2, b_col_off:b_col_off+2]
        # c = C[c_row_off:c_row_off+2, c_col_off:c_col_off+2]
        # Returns new col index
        c = col_base

        # m1 = (a00 + a11)(b00 + b11)
        U_mat[a_row_off + 0][c] = 1
        U_mat[a_row_off + 1][c] = 1
        V_mat[b_row_off + 0][c] = 1
        V_mat[b_row_off + 1][c] = 1
        W_mat[c_row_off + 0][c] = 1
        W_mat[c_row_off + 1][c] = 1
        c += 1

        # m2 = (a10 + a11) * b00
        U_mat[a_row_off + 1][c] = 1
        U_mat[a_row_off + 1][c] = 1 # wait, a10 and a11
        # Correction: m2 = (a10 + a11) b00
        U_mat[a_row_off + 1][c] = 1
        U_mat[a_row_off + 1][c] = 1 # This is wrong, need distinct indices? No, U is just coefficients.
        # U[0] = a00, U[1] = a10, U[2] = a01, U[3] = a11 ?
        # Let's define indices clearly.
        # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1) ?
        # Or row-major: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # The current program used: U[i*m + j]
        # So for a 2x2 block starting at (r0, c0):
        # idx 0: r0, c0
        # idx 1: r0, c0+1
        # idx 2: r0+1, c0
        # idx 3: r0+1, c0+1

        # Let's restart the Strassen implementation with clear indices
        # A block rows r0, r0+1; cols c0, c0+1
        # B block rows r0b, r0b+1; cols c0b, c0b+1
        # C block rows r0c, r0c+1; cols c0c, c0c+1

        # m1 = (a00 + a11)(b00 + b11)
        U_mat[a_row_off + 0][c] = 1  # a00
        U_mat[a_row_off + 3][c] = 1  # a11
        V_mat[b_row_off + 0][c] = 1  # b00
        V_mat[b_row_off + 3][c] = 1  # b11
        W_mat[c_row_off + 0][c] = 1  # c00 += m1
        W_mat[c_row_off + 3][c] = 1  # c11 += m1
        c += 1

        # m2 = (a10 + a11) b00
        U_mat[a_row_off + 2][c] = 1  # a10
        U_mat[a_row_off + 3][c] = 1  # a11
        V_mat[b_row_off + 0][c] = 1  # b00
        W_mat[c_row_off + 2][c] = 1  # c10 += m2
        W_mat[c_row_off + 3][c] = -1 # c11 -= m2
        c += 1

        # m3 = a00 (b01 - b11)
        U_mat[a_row_off + 0][c] = 1  # a00
        V_mat[b_row_off + 1][c] = 1  # b01
        V_mat[b_row_off + 3][c] = -1 # b11
        W_mat[c_row_off + 1][c] = 1  # c01 += m3
        W_mat[c_row_off + 3][c] = 1  # c11 += m3
        c += 1

        # m4 = a11 (b10 - b00)
        U_mat[a_row_off + 3][c] = 1  # a11
        V_mat[b_row_off + 2][c] = 1  # b10
        V_mat[b_row_off + 0][c] = -1 # b00
        W_mat[c_row_off + 0][c] = 1  # c00 += m4
        W_mat[c_row_off + 2][c] = 1  # c10 += m4
        c += 1

        # m5 = (a00 + a01) b11
        U_mat[a_row_off + 0][c] = 1  # a00
        U_mat[a_row_off + 1][c] = 1  # a01
        V_mat[b_row_off + 3][c] = 1  # b11
        W_mat[c_row_off + 0][c] = -1 # c00 -= m5
        W_mat[c_row_off + 1][c] = 1  # c01 += m5
        c += 1

        # m6 = (a10 - a00)(b00 + b01)
        U_mat[a_row_off + 2][c] = 1  # a10
        U_mat[a_row_off + 0][c] = -1 # a00
        V_mat[b_row_off + 0][c] = 1  # b00
        V_mat[b_row_off + 1][c] = 1  # b01
        W_mat[c_row_off + 3][c] = 1  # c11 += m6
        c += 1

        # m7 = (a01 - a11)(b10 + b11)
        U_mat[a_row_off + 1][c] = 1  # a01
        U_mat[a_row_off + 3][c] = -1 # a11
        V_mat[b_row_off + 2][c] = 1  # b10
        V_mat[b_row_off + 3][c] = 1  # b11
        W_mat[c_row_off + 0][c] = 1  # c00 += m7
        c += 1

        return c

    # Process Left Side (B cols 0,1)
    # A_top: rows 0,1 of A (global rows 0,1). Cols 0,1 -> A_TL. Cols 2,3 -> A_TR.
    # B_left: rows 0,1 -> B_TL. Rows 2,3 -> B_BL.
    # C_left: cols 0,1 of C.

    # Block 1: A_TL * B_TL
    # A_TL: A rows 0,1, cols 0,1 -> offset (0,0)
    # B_TL: B rows 0,1, cols 0,1 -> offset (0,0)
    # C_TL: C rows 0,1, cols 0,1 -> offset (0,0)
    col = strassen_block(U, V, W, 0, 0, 0, 0, 0, 0, col)

    # Block 2: A_TR * B_BL
    # A_TR: A rows 0,1, cols 2,3 -> offset (0,2)
    # B_BL: B rows 2,3, cols 0,1 -> offset (2,0)
    # C_TL: C rows 0,1, cols 0,1 -> offset (0,0)
    # Note: B_BL is B[2:4, 0:2].
    col = strassen_block(U, V, W, 0, 2, 2, 0, 0, 0, col)

    # Bottom Part Left: A_bottom (row 2) * B_left (4x2)
    # A_bottom: A row 2, cols 0..3 -> offset (2,0) but it's 1x4.
    # We need 4 rank-1 terms.
    # C_bottom: C row 2, cols 0,1.
    # Term 1: A[2,0] * B[0,0] -> C[0,0]
    U[2*4 + 0][col] = 1
    V[0*4 + 0][col] = 1
    W[0*3 + 2][col] = 1
    col += 1
    # Term 2: A[2,0] * B[0,1] -> C[0,1]
    U[2*4 + 0][col] = 1
    V[0*4 + 1][col] = 1
    W[1*3 + 2][col] = 1
    col += 1
    # Term 3: A[2,1] * B[1,0] -> C[0,0]
    U[2*4 + 1][col] = 1
    V[1*4 + 0][col] = 1
    W[0*3 + 2][col] = 1
    col += 1
    # Term 4: A[2,1] * B[1,1] -> C[0,1]
    U[2*4 + 1][col] = 1
    V[1*4 + 1][col] = 1
    W[1*3 + 2][col] = 1
    col += 1
    # Term 5: A[2,2] * B[2,0] -> C[0,0]
    U[2*4 + 2][col] = 1
    V[2*4 + 0][col] = 1
    W[0*3 + 2][col] = 1
    col += 1
    # Term 6: A[2,2] * B[2,1] -> C[0,1]
    U[2*4 + 2][col] = 1
    V[2*4 + 1][col] = 1
    W[1*3 + 2][col] = 1
    col += 1
    # Term 7: A[2,3] * B[3,0] -> C[0,0]
    U[2*4 + 3][col] = 1
    V[3*4 + 0][col] = 1
    W[0*3 + 2][col] = 1
    col += 1
    # Term 8: A[2,3] * B[3,1] -> C[0,1]
    U[2*4 + 3][col] = 1
    V[3*4 + 1][col] = 1
    W[1*3 + 2][col] = 1
    col += 1

    # Wait, my "Bottom Part" rank calculation was wrong.
    # A_bottom is 1x4. B_left is 4x2.
    # The product is 1x2.
    # The rank of a 1x4x2 tensor is max(4, 2, 1) ? No.
    # The tensor is T[i,j,k] = A[i,j] B[j,k].
    # Here i is fixed (row 2 of A, row 2 of C).
    # j ranges 0..3. k ranges 0..1.
    # This is just the matrix multiplication of a row vector and a matrix.
    # The tensor is rank 4?
    # Let's check.
    # C[2,0] = sum_j A[2,j] B[j,0]
    # C[2,1] = sum_j A[2,j] B[j,1]
    # We can write this as:
    # M_j = A[2,j]
    # Then C[2,0] = sum M_j B[j,0]
    # C[2,1] = sum M_j B[j,1]
    # This requires 4 independent products M_j * B[j,0] and M_j * B[j,1]?
    # No, we can't reuse M_j for different B columns if they are in separate rank-1 terms.
    # A rank-1 term is u_i v_j w_k.
    # If we use M_j as the "u" part, it's fixed.
    # We need to produce two outputs.
    # Term 1: u = A[2,0], v = B[0,0], w = 1 for C[0]
    # Term 2: u = A[2,0], v = B[0,1], w = 1 for C[1]
    # These are distinct rank-1 tensors because v and w are different.
    # So yes, it takes 4 (for j) * 2 (for k) = 8 terms?
    # Let's re-evaluate rank of 1x4x2 multiplication.
    # The tensor is sum_j e_j (A part) tensor B_j (B part) tensor (output part).
    # Actually, it's just a linear map.
    # The rank of the matrix multiplication tensor [1, 4, 2] is 4.
    # Why?
    # C[0] = A[0]B[00] + A[1]B[10] + A[2]B[20] + A[3]B[30]
    # C[1] = A[0]B[01] + A[1]B[11] + A[2]B[21] + A[3]B[31]
    # We can group by A:
    # M0 = A[0]. Terms: M0*B[00] -> C0, M0*B[01] -> C1.
    # These are two separate rank 1 terms because the output vectors are different (e0 vs e1).
    # So for each j, we have 2 terms. Total 8 terms.
    # So rank is 8.
    # My previous "rank 4" was incorrect. It would be 4 if we were computing 1x4x1.
    # So Left Side: 14 (Strassen) + 8 (Bottom) = 22.
    # Right Side: 22.
    # Total: 44.
    # This is the same as the original solution!
    # The original solution tiled 4 blocks of 3x2x2.
    # 3x2x2 rank is 11?
    # If 3x2x2 is 11, then 44 is correct.
    # If I split into 2x4x2 + 1x4x2, I get 14 + 8 = 22 per side.
    # 22 * 2 = 44.
    # So rank 44 is confirmed.
    # Can we do better?
    # Maybe the 3x4x2 block has rank < 14+8=22?
    # Rank of 3x4x2 tensor.
    # Lower bound is max(3*4, 4*2, 3*2) = 12.
    # Strassen based: 2x2x2 (7) + 1x2x2 (4) = 11 for 3x2x2.
    # So 3x4x2 = 3x2x2 + 3x2x2 = 11 + 11 = 22.
    # It seems 44 is the best we can get with this block structure.
    # The original solution is already optimal for this structure.
    # Let's stick with the original solution's logic but maybe clean it up?
    # Or just return the original solution?
    # The prompt asks to "Rewrite the program from scratch with a stronger algorithm".
    # If I can't find a stronger one, I should return a valid one.
    # But I should try to see if there's any overlap or optimization.
    # The original solution:
    # for b_r in range(2):
    #   for b_c in range(2):
    #     Strassen (7)
    #     Trivial (4)
    # Total 11 per block. 4 blocks -> 44.
    # My analysis:
    # Left (cols 0,1):
    #   Top (rows 0,1): 2x4x2.
    #   Split A into 2x2 + 2x2. B into 2x2 + 2x2.
    #   A_TL B_TL (7) + A_TR B_BL (7) = 14.
    #   Bot (row 2): 1x4x2.
    #   Rank 8.
    #   Total 22.
    # Right (cols 2,3):
    #   Same 22.
    # Total 44.
    #
    # Is there a way to do 3x4x2 with rank < 22?
    # If we use a better algorithm for 3x4x2.
    # Maybe we can mix the "bottom" row with the "top" rows in a smarter way?
    # Strassen is for 2x2.
    # What if we use a 3x3 Strassen? No, B is 4 wide.
    #
    # Let's just implement the rank 44 solution cleanly.
    # It is the same as the current best score (1.0909).
    # If I return exactly the same rank, the score will be the same.
    # Is there any way to reduce it?
    # Maybe 3x4x4 has a rank < 44?
    # The trivial rank is 48.
    # 44 is an improvement.
    # If I can't find < 44, I will return 44.
    # I will implement the 3x4x2 split version as it is conceptually cleaner?
    # Or the block tiling version?
    # The block tiling version is what the current best uses.
    # I will rewrite it to be sure it is correct and maybe slightly more efficient in code.

    # Reset
    rank = 44
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0

    # Block tiling: 2x2 blocks in A rows and B cols.
    # A is 3x4. B is 4x4.
    # Blocks of A: (2x2) at (0,0), (2x2) at (0,2), (1x2) at (2,0), (1x2) at (2,2).
    # Blocks of B: (4x2) at (0,0), (4x2) at (0,2).
    # Actually, the original code iterated b_r (A row block) and b_c (B col block).
    # b_r in 0..1, b_c in 0..1.
    # Inside:
    # Strassen for A[2x2] * B[2x2] part.
    # But B is 4x4. The Strassen part in the original code used B[2x2] sub-blocks.
    # Let's trace original code:
    # U[0*m + (b_r*2+0)][col] = 1 ...
    # This accesses A row b_r*2, b_r*2+1.
    # V[(b_r*2+0)*p + (b_c*2+0)][col] = 1 ...
    # This accesses B row b_r*2, b_r*2+1. Col b_c*2, b_c*2+1.
    # So it computes A[2x2] * B[2x2] for each combination of row block and col block?
    # No, that would be A[2x2] * B[2x2] -> C[2x2].
    # But A is 3x4.
    # If we tile A into 2x2 blocks, we have row 0,1 and row 2 (leftover).
    # The original code seems to handle the "leftover" row separately in a "Trivial part".
    #
    # Let's stick to the 3x4x2 split (rank 22 per side) as it is easier to verify.
    # Left Side (cols 0,1 of C):
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2] (Strassen, 7)
    # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2] (Strassen, 7)
    # 3. A[2:3, 0:4] * B[0:4, 0:2] -> C[2:3, 0:2] (Trivial, 8)
    # Total 22.
    # Right Side (cols 2,3 of C):
    # 1. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4] (Strassen, 7)
    # 2. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4] (Strassen, 7)
    # 3. A[2:3, 0:4] * B[0:4, 2:4] -> C[2:3, 2:4] (Trivial, 8)
    # Total 22.
    # Grand Total 44.

    # Implementation:

    # Helper to add a Strassen block
    def add_strassen(a_r, a_c, b_r, b_c, c_r, c_c, start_col):
        c = start_col
        # Indices relative to start
        # A: (a_r, a_c) to (a_r+1, a_c+1)
        # B: (b_r, b_c) to (b_r+1, b_c+1)
        # C: (c_r, c_c) to (c_r+1, c_c+1)

        # m1
        U[a_r*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + c_r][c] += 1
        W[(c_c+1)*n + (c_r+1)][c] += 1
        c += 1

        # m2
        U[(a_r+1)*m + a_c][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[b_r*p + b_c][c] += 1
        W[c_c*n + (a_r+1)][c] += 1
        W[(c_c+1)*n + (a_r+1)][c] -= 1
        c += 1

        # m3
        U[a_r*m + a_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] -= 1
        W[(c_c+1)*n + a_r][c] += 1
        W[(c_c+1)*n + (a_r+1)][c] += 1
        c += 1

        # m4
        U[(a_r+1)*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + b_c][c] += 1
        V[b_r*p + b_c][c] -= 1
        W[c_c*n + a_r][c] += 1
        W[c_c*n + (a_r+1)][c] += 1
        c += 1

        # m5
        U[a_r*m + a_c][c] += 1
        U[a_r*m + (a_c+1)][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + a_r][c] -= 1
        W[(c_c+1)*n + a_r][c] += 1
        c += 1

        # m6
        U[(a_r+1)*m + a_c][c] += 1
        U[a_r*m + a_c][c] -= 1
        V[b_r*p + b_c][c] += 1
        V[b_r*p + (b_c+1)][c] += 1
        W[(c_c+1)*n + (a_r+1)][c] += 1
        c += 1

        # m7
        U[a_r*m + (a_c+1)][c] += 1
        U[(a_r+1)*m + (a_c+1)][c] -= 1
        V[(b_r+1)*p + b_c][c] += 1
        V[(b_r+1)*p + (b_c+1)][c] += 1
        W[c_c*n + a_r][c] += 1
        c += 1

        return c

    # Helper to add trivial 1x4x2 block
    # A row a_r, cols 0..3. B rows 0..3, cols b_c, b_c+1.
    # C row a_r, cols b_c, b_c+1.
    def add_trivial_1x4x2(a_r, b_c, start_col):
        c = start_col
        for j in range(4):
            # Term for col b_c
            U[a_r*m + j][c] += 1
            V[j*p + b_c][c] += 1
            W[b_c*n + a_r][c] += 1
            c += 1
            # Term for col b_c+1
            U[a_r*m + j][c] += 1
            V[j*p + (b_c+1)][c] += 1
            W[(b_c+1)*n + a_r][c] += 1
            c += 1
        return c

    # Left Side (b_c = 0)
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_strassen(0, 0, 0, 0, 0, 0, col)
    # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    col = add_strassen(0, 2, 2, 0, 0, 0, col)
    # 3. A[2:3, 0:4] * B[0:4, 0:2] -> C[2:3, 0:2]
    col = add_trivial_1x4x2(2, 0, col)

    # Right Side (b_c = 2)
    # 1. A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    col = add_strassen(0, 0, 0, 2, 0, 2, col)
    # 2. A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    col = add_strassen(0, 2, 2, 2, 0, 2, col)
    # 3. A[2:3, 0:4] * B[0:4, 2:4] -> C[2:3, 2:4]
    col = add_trivial_1x4x2(2, 2, col)

    return {"U": U, "V": V, "W": W}
