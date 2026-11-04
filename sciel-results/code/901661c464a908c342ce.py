"""Matrix multiplication 4x4x8: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8

    # Strategy: Use Strassen's rank-7 algorithm for 2x2x2 block multiplication.
    # Partition A (4x4) into 2x2 blocks of 2x2 matrices:
    #   A = [[A11, A12], [A21, A22]]
    # Partition B (4x8) into 2x4 blocks of 2x2 matrices:
    #   B = [[B11, B12, B13, B14], [B21, B22, B23, B24]]
    # Result C (4x8) into 2x4 blocks of 2x2 matrices:
    #   C = [[C11, C12, C13, C14], [C21, C22, C23, C24]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C13 = A11*B13 + A12*B23
    # C14 = A11*B14 + A12*B24
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    # C23 = A21*B13 + A22*B23
    # C24 = A21*B14 + A22*B24
    #
    # Each Cij = Ai1*B1j + Ai2*B2j is a sum of two 2x2 matrix products.
    # We can use Strassen to compute each pair more efficiently.
    #
    # Actually, let's think about this differently. The full problem is 4x4 * 4x8.
    # We can view this as: for each row-block of A (2 rows) and each column-block of B (2 cols),
    # we compute a 2x2 output block.
    #
    # Better approach: Use the tensor product of Strassen with the schoolbook for the "width" dimension.
    # 
    # Alternative: Direct Strassen-like approach for 4x4x4, then extend to 4x4x8.
    #
    # For 4x4x4: partition into 2x2 blocks of 2x2 matrices.
    # C[i,j] = sum_k A[i,k]*B[k,j]
    # Using Strassen at the block level:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2 matrix product, which itself can be done with Strassen (rank 7).
    # Total rank for 4x4x4: 7 * 7 = 49.
    #
    # For 4x4x8: We can view B as two 4x4 matrices side by side: B = [B_left, B_right].
    # Then C = A * B = [A*B_left, A*B_right].
    # Each A*B_XX is a 4x4x4 problem with rank 49.
    # Total rank: 49 + 49 = 98.
    #
    # This gives rank 98, which is an improvement over 128.

    rank = 98

    U = np.zeros((n * m, rank))  # 16 x 98
    V = np.zeros((m * p, rank))  # 32 x 98
    W = np.zeros((p * n, rank))  # 32 x 98

    # Helper: add a Strassen 2x2x2 decomposition for a specific block
    # The Strassen 7 products for 2x2 matrices:
    # For matrices X (2x2), Y (2x2) -> Z = X*Y (2x2)
    # M1 = (X00+X11)*(Y00+Y11)
    # M2 = (X10+X11)*Y00
    # M3 = X00*(Y01-Y11)
    # M4 = X11*(Y10-Y00)
    # M5 = (X00+X01)*Y11
    # M6 = (X10-X00)*(Y00+Y01)
    # M7 = (X01-X11)*(Y10+Y11)
    #
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6

    # For 4x4x4 using Strassen at block level (2x2 blocks of 2x2):
    # We need to decompose the 4x4x4 tensor into 49 rank-1 terms.
    #
    # Let's build it step by step.
    # Block indices: A has blocks A00, A01, A10, A11 (each 2x2)
    # B has blocks B00, B01, B10, B11 (each 2x2)
    # C has blocks C00, C01, C10, C11 (each 2x2)
    #
    # The 7 Strassen combinations at the 4x4 block level:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # Each Mi is a 2x2 * 2x2 = 2x2 product, decomposed with Strassen (7 terms each).
    # Total: 7 * 7 = 49 terms.

    # Indexing for 2x2 blocks within 4x4:
    # A[i,j] where i,j in 0..3. Block A00: i in {0,1}, j in {0,1}
    # Block A01: i in {0,1}, j in {2,3}
    # Block A10: i in {2,3}, j in {0,1}
    # Block A11: i in {2,3}, j in {2,3}
    #
    # Similarly for B (4x4) and C (4x4).
    # For B (4x8), we split into B_left (4x4, cols 0-3) and B_right (4x4, cols 4-7).

    # Strassen 2x2x2 coefficients:
    # For product Z = X * Y where X,Y are 2x2:
    # The 7 rank-1 terms have the following structure:
    # Each term contributes to Z[i,j] = sum_t U_X[i,j,t] * U_Y[i,j,t] * U_Z[i,j,t]
    # (using appropriate indexing)

    # Let me define the Strassen 2x2x2 decomposition explicitly.
    # X = [[x00, x01], [x10, x11]]
    # Y = [[y00, y01], [y10, y11]]
    # Z = X*Y = [[z00, z01], [z10, z11]]
    #
    # M1 = (x00+x11)*(y00+y11)
    # M2 = (x10+x11)*y00
    # M3 = x00*(y01-y11)
    # M4 = x11*(y10-y00)
    # M5 = (x00+x01)*y11
    # M6 = (x10-x00)*(y00+y01)
    # M7 = (x01-x11)*(y10+y11)
    #
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6

    # For each of the 7 Strassen products at the 4x4 level, we need to expand
    # the 2x2 * 2x2 product using Strassen again.

    # Let's build the full decomposition.
    # For the 4x4x4 case:
    # A is 4x4, B is 4x4, C is 4x4.
    # U_A has shape (16, 49), U_B has shape (16, 49), U_C has shape (16, 49).

    # Strassen 2x2x2: for X (2x2) * Y (2x2) = Z (2x2)
    # The 7 terms: each term t has coefficients for X elements, Y elements, Z elements.
    # X elements indexed as (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # Same for Y and Z.

    # Term 1: (x00+x11)*(y00+y11) -> contributes to z00, z11
    # x: [1, 0, 0, 1]
    # y: [1, 0, 0, 1]
    # z: [1, 0, 0, 1]  (z00 and z11)
    
    # Term 2: (x10+x11)*y00 -> contributes to z10, z11
    # x: [0, 0, 1, 1]
    # y: [1, 0, 0, 0]
    # z: [0, 0, 1, 1]  (z10 and z11)
    
    # Term 3: x00*(y01-y11) -> contributes to z01, z11
    # x: [1, 0, 0, 0]
    # y: [0, 1, 0, -1]
    # z: [0, 1, 0, 1]  (z01 and z11)
    
    # Term 4: x11*(y10-y00) -> contributes to z00, z10
    # x: [0, 0, 0, 1]
    # y: [-1, 0, 1, 0]
    # z: [1, 0, 1, 0]  (z00 and z10)
    
    # Term 5: (x00+x01)*y11 -> contributes to z00, z01
    # x: [1, 1, 0, 0]
    # y: [0, 0, 0, 1]
    # z: [1, 1, 0, 0]  (z00 and z01)
    
    # Term 6: (x10-x00)*(y00+y01) -> contributes to z10, z11
    # x: [-1, 0, 1, 0]
    # y: [1, 1, 0, 0]
    # z: [0, 0, 1, 1]  (z10 and z11)
    
    # Term 7: (x01-x11)*(y10+y11) -> contributes to z00, z01
    # x: [0, 1, 0, -1]
    # y: [0, 0, 1, 1]
    # z: [1, 1, 0, 0]  (z00 and z01)

    strassen_x = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1],  # M7
    ]
    strassen_y = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1],   # M7
    ]
    strassen_z = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [1, 1, 0, 0],   # M5
        [0, 0, 1, 1],   # M6
        [1, 1, 0, 0],   # M7
    ]

    # Now for the 4x4x4 Strassen at block level:
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # B blocks: B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    # C blocks: C00 (rows 0-1, cols 0-1), C01 (rows 0-1, cols 2-3), C10 (rows 2-3, cols 0-1), C11 (rows 2-3, cols 2-3)

    # The 7 Strassen combinations:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)

    # For each Mi, the "X" matrix is a 2x2 block of A (or sum/diff of blocks),
    # and the "Y" matrix is a 2x2 block of B (or sum/diff of blocks).
    # The result contributes to C blocks.

    # Let's define the block-level Strassen terms:
    # Each term specifies:
    # - Which A blocks and with what signs
    # - Which B blocks and with what signs
    # - Which C blocks it contributes to and with what signs

    # Block A indices: 0=A00, 1=A01, 2=A10, 3=A11
    # Block B indices: 0=B00, 1=B01, 2=B10, 3=B11
    # Block C indices: 0=C00, 1=C01, 2=C10, 3=C11

    # M1 = (A00+A11)*(B00+B11)
    # A blocks: {0: +1, 3: +1}
    # B blocks: {0: +1, 3: +1}
    # C blocks: contributes to C00, C11 (from Strassen z for M1: [1,0,0,1])
    # Wait, the C block contributions depend on which Strassen z term.
    # For M1 at the 4x4 level, the result is a 2x2 matrix (one 2x2 block of C? No...)
    #
    # Actually, I need to be more careful. At the 4x4 block level:
    # M1 = (A00+A11)*(B00+B11) is a 2x2 matrix product giving a 2x2 result.
    # But which 2x2 block of C does it contribute to?
    #
    # C00 = M1 + M4 - M5 + M7 (where each Mi is a 2x2 matrix)
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # So M1 (a 2x2 matrix) contributes to both C00 and C11.
    # Specifically, the 2x2 result of M1 adds to C00 (with coeff +1) and C11 (with coeff +1).
    #
    # So for each of the 7 block-level Strassen terms, the 2x2 result is distributed
    # among C blocks according to the Strassen z coefficients.

    # Block-level Strassen terms (for 4x4x4):
    # term_idx: (A_blocks_with_signs, B_blocks_with_signs, C_block_contributions)
    # C_block_contributions: dict from C_block_idx to coefficient

    block_strassen = [
        # M1 = (A00+A11)*(B00+B11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),  # C00: +1, C11: +1
        # M2 = (A10+A11)*B00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: 1}),  # C10: +1, C11: +1
        # M3 = A00*(B01-B11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),  # C01: +1, C11: +1
        # M4 = A11*(B10-B00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),  # C00: +1, C10: +1
        # M5 = (A00+A01)*B11
        ({0: 1, 1: 1}, {3: 1}, {0: 1, 1: 1}),  # C00: +1, C01: +1
        # M6 = (A10-A00)*(B00+B01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {2: 1, 3: 1}),  # C10: +1, C11: +1
        # M7 = (A01-A11)*(B10+B11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1, 1: 1}),  # C00: +1, C01: +1
    ]

    # Now, for each block-level Strassen term, we expand the 2x2*2x2 product
    # using the inner Strassen (7 terms).
    #
    # For a given block-level term with A_blocks and B_blocks:
    # The "X" matrix (2x2) has elements from the specified A blocks.
    # The "Y" matrix (2x2) has elements from the specified B blocks.
    # The inner Strassen gives 7 rank-1 terms for the 2x2*2x2 product.
    #
    # Each inner Strassen term t has:
    # - x_coeffs: 4 values for the 4 elements of the 2x2 X matrix
    # - y_coeffs: 4 values for the 4 elements of the 2x2 Y matrix
    # - z_coeffs: 4 values for the 4 elements of the 2x2 Z matrix
    #
    # The 2x2 X matrix elements correspond to specific (i,j) positions in A.
    # For A block b (0=A00, 1=A01, 2=A10, 3=A11):
    #   Element (0,0) of block b -> A[2*(b//2), 2*(b%2)]
    #   Element (0,1) of block b -> A[2*(b//2), 2*(b%2)+1]
    #   Element (1,0) of block b -> A[2*(b//2)+1, 2*(b%2)]
    #   Element (1,1) of block b -> A[2*(b//2)+1, 2*(b%2)+1]
    #
    # Wait, that's not right. Let me re-index.
    # A is 4x4. Block A00: rows 0-1, cols 0-1.
    #   A00[0,0] = A[0,0], A00[0,1] = A[0,1], A00[1,0] = A[1,0], A00[1,1] = A[1,1]
    # Block A01: rows 0-1, cols 2-3.
    #   A01[0,0] = A[0,2], A01[0,1] = A[0,3], A01[1,0] = A[1,2], A01[1,1] = A[1,3]
    # Block A10: rows 2-3, cols 0-1.
    #   A10[0,0] = A[2,0], A10[0,1] = A[2,1], A10[1,0] = A[3,0], A10[1,1] = A[3,1]
    # Block A11: rows 2-3, cols 2-3.
    #   A11[0,0] = A[2,2], A11[0,1] = A[2,3], A11[1,0] = A[3,2], A11[1,1] = A[3,3]

    def block_element_pos(block_idx, local_i, local_j):
        """Return (row, col) in the 4x4 matrix for block block_idx, local position (local_i, local_j)."""
        block_row = block_idx // 2  # 0 or 1
        block_col = block_idx % 2   # 0 or 1
        return (2 * block_row + local_i, 2 * block_col + local_j)

    # For B (4x4 in the 4x4x4 case), same indexing.
    # For C (4x4 in the 4x4x4 case), same indexing.
    # But for the full 4x4x8 problem, B is 4x8 and C is 4x8.
    # We split B into B_left (4x4, cols 0-3) and B_right (4x4, cols 4-7).
    # Similarly C into C_left (4x4, cols 0-3) and C_right (4x4, cols 4-7).

    # Let's build the 4x4x4 decomposition first, then duplicate for the two halves.

    def build_4x4x4_decomp():
        """Build rank-49 decomposition for 4x4 * 4x4 = 4x4."""
        U_A = np.zeros((16, 49))
        U_B = np.zeros((16, 49))
        U_C = np.zeros((16, 49))

        col = 0
        for bs_idx in range(7):
            A_blocks = block_strassen[bs_idx][0]
            B_blocks = block_strassen[bs_idx][1]
            C_contrib = block_strassen[bs_idx][2]

            for inner_t in range(7):
                x_coeffs = strassen_x[inner_t]
                y_coeffs = strassen_y[inner_t]
                z_coeffs = strassen_z[inner_t]

                # For each element of the 2x2 X matrix (local position li, lj):
                # The coefficient is x_coeffs[li*2+lj]
                # This element comes from A blocks in A_blocks with the specified signs.
                # But wait, the X matrix is a 2x2 matrix formed by combining A blocks.
                # For example, for M1: X = A00 + A11.
                # X[0,0] = A00[0,0] + A11[0,0] = A[0,0] + A[2,2]
                # X[0,1] = A00[0,1] + A11[0,1] = A[0,1] + A[2,3]
                # etc.
                #
                # So for each local position (li, lj) in the 2x2 X matrix:
                # The contribution to U_A at column col is:
                #   sum over (block, sign) in A_blocks: sign * (1 if this is the element at (li,lj) in that block)
                # But actually, each element of X is a linear combination of A elements.
                # X[li, lj] = sum_{block in A_blocks} sign * A_element_at(block, li, lj)
                #
                # So U_A[A_element_index][col] += x_coeffs[li*2+lj] * sign
                # for each (block, sign) in A_blocks and each (li, lj).

                for (block, sign) in A_blocks.items():
                    for li in range(2):
                        for lj in range(2):
                            row, col_a = block_element_pos(block, li, lj)
                            a_idx = row * 4 + col_a  # A is 4x4, so index = row*4+col
                            U_A[a_idx, col] += x_coeffs[li * 2 + lj] * sign

                for (block, sign) in B_blocks.items():
                    for li in range(2):
                        for lj in range(2):
                            row, col_b = block_element_pos(block, li, lj)
                            b_idx = row * 4 + col_b  # B is 4x4, so index = row*4+col
                            U_B[b_idx, col] += y_coeffs[li * 2 + lj] * sign

                # For C: the result Z (2x2) is distributed among C blocks.
                # Z[li, lj] contributes to C blocks with the specified coefficients.
                for (c_block, c_sign) in C_contrib.items():
                    for li in range(2):
                        for lj in range(2):
                            row, col_c = block_element_pos(c_block, li, lj)
                            c_idx = col_c * 4 + row  # C is 4x4, W index = k*n+i = col*4+row
                            U_C[c_idx, col] += z_coeffs[li * 2 + lj] * c_sign

                col += 1

        return U_A, U_B, U_C

    U_A_44, U_B_44, U_C_44 = build_4x4x4_decomp()

    # Now extend to 4x4x8:
    # B is 4x8, split into B_left (cols 0-3) and B_right (cols 4-7).
    # C is 4x8, split into C_left (cols 0-3) and C_right (cols 4-7).
    #
    # C_left = A * B_left (4x4x4 problem, rank 49)
    # C_right = A * B_right (4x4x4 problem, rank 49)
    #
    # Total rank: 98.

    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))

    # First half: B_left (cols 0-3), C_left (cols 0-3)
    U[:, 0:49] = U_A_44
    V[0:16, 0:49] = U_B_44  # B_left: rows 0-3, cols 0-3 -> indices 0..15
    # C_left: W index = k*4+i where k in 0..3, i in 0..3 -> indices 0..15
    W[0:16, 0:49] = U_C_44

    # Second half: B_right (cols 4-7), C_right (cols 4-7)
    U[:, 49:98] = U_A_44
    # B_right: rows 0-3, cols 4-7 -> indices: row*8+col where col in 4..7
    # So B_right index = row*8 + (4 + col_local) where col_local in 0..3
    # U_B_44 has indices row*4+col_local. We need to map to row*8+4+col_local.
    for r in range(4):
        for c in range(4):
            V[r * 8 + 4 + c, 49:98] = U_B_44[r * 4 + c, :]

    # C_right: W index = k*4+i where k in 4..7, i in 0..3
    # U_C_44 has indices col_local*4+row. We need to map to (4+col_local)*4+row.
    for k in range(4):
        for i in range(4):
            W[(4 + k) * 4 + i, 49:98] = U_C_44[k * 4 + i, :]

    # Convert to lists of lists with integer entries
    U_list = [[int(round(x)) for x in row] for row in U]
    V_list = [[int(round(x)) for x in row] for row in V]
    W_list = [[int(round(x)) for x in row] for row in W]

    return {"U": U_list, "V": V_list, "W": W_list}
