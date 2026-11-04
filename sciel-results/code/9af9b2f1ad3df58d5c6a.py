"""Matrix multiplication 2x3x8 rank-minimal decomposition."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # We decompose the 2x3x8 multiplication into sub-blocks using the additive property of tensor rank.
    #
    # Let A be 2x3, B be 3x8. The product C is 2x8.
    # We can view the columns of A in groups.
    #
    # Strategy:
    # Split the 3 columns of A into a 2-column part and a 1-column part.
    # A = [A_2 | A_1] where A_2 is 2x2 and A_1 is 2x1.
    # B = [B_2; B_1] where B_2 is 2x8 and B_1 is 1x8.
    # C = A_2 * B_2 + A_1 * B_1.
    #
    # The term A_2 * B_2 is a 2x2 * 2x8 multiplication.
    # The term A_1 * B_1 is a 2x1 * 1x8 multiplication.
    #
    # For the 2x2 * 2x8 part:
    #   We can view this as two independent 2x2 * 2x4 multiplications? No, that's not right.
    #   Actually, 2x2 * 2x8 is equivalent to multiplying a 2x2 matrix by each of the 8 columns.
    #   But we can group the 8 columns into four groups of 2 columns.
    #   So it's four independent 2x2 * 2x2 multiplications.
    #   Each 2x2 * 2x2 can be done with rank 7 (Strassen).
    #   So 4 * 7 = 28 for the 2x2 * 2x8 part.
    #
    # For the 2x1 * 1x8 part:
    #   This is an outer product of a 2x1 vector and a 1x8 vector.
    #   This is rank 8 (each entry is a separate multiplication).
    #   Wait, no. A_1 is 2x1, B_1 is 1x8. The product is 2x8.
    #   C[i,k] = A_1[i,0] * B_1[0,k].
    #   This is 2*8 = 16 scalar multiplications.
    #   But we can do better. The tensor for 2x1x1 * 1x1x8 is the sum of 16 rank-1 tensors.
    #   Actually, it's just 16 independent multiplications, so rank 16.
    #
    # Total rank = 28 + 16 = 44.
    #
    # Can we do better?
    # Let's try splitting the 3 columns differently.
    # What if we split into three 1-column parts?
    # A = [A_1 | A_2 | A_3], each 2x1.
    # B = [B_1; B_2; B_3], each 1x8.
    # C = A_1*B_1 + A_2*B_2 + A_3*B_3.
    # Each term is rank 16. Total rank = 48. Worse.
    #
    # What about splitting the 8 columns?
    # A is 2x3, B is 3x8.
    # Split B into four 3x2 blocks.
    # C = A*B_1 + A*B_2 + A*B_3 + A*B_4.
    # Each A*B_i is 2x3 * 3x2.
    # The rank of 2x3x2 multiplication...
    # 2x3*3x2: A is 2x3, B is 3x2.
    # We can split A into two 2x2 and 2x1? No, A has 3 columns.
    # Actually, 2x3*3x2 can be viewed as:
    #   Split the 3 columns of A into 2 and 1.
    #   Split the 2 rows of B into 2 and... no, B has 3 rows.
    #   Let's use the known result: R(2x3, 3x2) = 12? Or can we do better?
    #   Schoolbook is 2*3*2 = 12.
    #   Can we use Strassen-like tricks?
    #   The tensor is 2x3x2.
    #   We can write it as a sum of rank-1 tensors.
    #   Actually, I recall that R(2x3,3x2) might be 12, but let's check if we can reduce it.
    #   Consider the 2x3 matrix A and 3x2 matrix B.
    #   C = A*B.
    #   C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    #   C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    #   C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    #   C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    #
    #   This is 12 multiplications. Can we do it in fewer?
    #   I don't think so for 2x3x2, but let's assume rank 12 for now.
    #   Then 4 * 12 = 48. Same as schoolbook.
    #
    # Let's go back to the 44-rank construction.
    #
    # Construction details:
    # A is 2x3: indices (i, j) where i in {0,1}, j in {0,1,2}
    # B is 3x8: indices (j, k) where j in {0,1,2}, k in {0,...,7}
    # C is 2x8: indices (i, k) where i in {0,1}, k in {0,...,7}
    #
    # Split: j in {0,1} for A_2 and B_2, j=2 for A_1 and B_1.
    #
    # Part 1: A_2 (2x2) * B_2 (2x8)
    #   B_2 has 8 columns. Split into 4 blocks of 2 columns each.
    #   Block b (b=0,1,2,3): columns 2b, 2b+1 of B_2.
    #   Each block is a 2x2 matrix.
    #   For each block, use Strassen's algorithm (rank 7).
    #
    #   Strassen for 2x2*2x2:
    #   Let X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]
    #   Z = X*Y
    #   m1 = (x00+x11)*(y00+y11)
    #   m2 = (x10+x11)*y00
    #   m3 = x00*(y01-y11)
    #   m4 = x11*(y10-y00)
    #   m5 = (x00+x01)*y11
    #   m6 = (x10-x00)*(y00+y01)
    #   m7 = (x01-x11)*(y10+y11)
    #   z00 = m1 + m4 - m5 + m7
    #   z01 = m3 + m5
    #   z10 = m2 + m4
    #   z11 = m1 - m2 + m3 + m6
    #
    #   In tensor notation, the 7 rank-1 terms are:
    #   (x00+x11) * (y00+y11) * (z00 + z11)  -- wait, no.
    #   Each m_i is a product of a linear form in X and a linear form in Y.
    #   And z_ij is a linear combination of the m's.
    #
    #   So for each block, we add 7 rank-1 tensors.
    #   Total for Part 1: 4 * 7 = 28.
    #
    # Part 2: A_1 (2x1) * B_1 (1x8)
    #   A_1 = [a0, a1]^T, B_1 = [b0, b1, ..., b7]
    #   C_part = [[a0*b0, a0*b1, ..., a0*b7],
    #             [a1*b0, a1*b1, ..., a1*b7]]
    #   This is 16 scalar multiplications.
    #   Each is a rank-1 tensor.
    #   Total for Part 2: 16.
    #
    # Grand total: 28 + 16 = 44.
    #
    # Now, let's construct the U, V, W matrices.
    # U has shape (n*m, rank) = (2*3, 44) = (6, 44)
    # V has shape (m*p, rank) = (3*8, 44) = (24, 44)
    # W has shape (p*n, rank) = (8*2, 44) = (16, 44)
    #
    # Indexing:
    # U[i*m + j, c] where i in 0..1, j in 0..2
    # V[j*p + k, c] where j in 0..2, k in 0..7
    # W[k*n + i, c] where k in 0..7, i in 0..1
    #
    # Part 1: 28 columns (c = 0..27)
    #   For block b (0..3), Strassen term s (0..6):
    #     c = b*7 + s
    #
    #   Strassen terms for a 2x2*2x2 with X (2x2) and Y (2x2):
    #   X indices: (x_row, x_col) where x_row, x_col in {0,1}
    #   Y indices: (y_row, y_col) where y_row, y_col in {0,1}
    #   Z indices: (z_row, z_col) where z_row, z_col in {0,1}
    #
    #   In our case, X = A_2, so x_row = i (0..1), x_col = j (0..1)
    #   Y = B_2 block b, so y_row = j (0..1), y_col = k_local (0..1) where k_global = 2*b + k_local
    #   Z = C block, so z_row = i (0..1), z_col = k_global (0..7)
    #
    #   Strassen coefficients:
    #   m1 = (x00+x11)*(y00+y11) -> contributes to z00 and z11
    #   m2 = (x10+x11)*y00 -> contributes to z10 and z11
    #   m3 = x00*(y01-y11) -> contributes to z01 and z11
    #   m4 = x11*(y10-y00) -> contributes to z10 and z11
    #   m5 = (x00+x01)*y11 -> contributes to z00 and z01
    #   m6 = (x10-x00)*(y00+y01) -> contributes to z11
    #   m7 = (x01-x11)*(y10+y11) -> contributes to z00
    #
    #   Let's define the 7 terms explicitly:
    #   Term 0: m1 = (x00+x11)*(y00+y11)
    #     U: x00->1, x11->1
    #     V: y00->1, y11->1
    #     W: z00->1, z11->1
    #
    #   Term 1: m2 = (x10+x11)*y00
    #     U: x10->1, x11->1
    #     V: y00->1
    #     W: z10->1, z11->(-1)
    #
    #   Term 2: m3 = x00*(y01-y11)
    #     U: x00->1
    #     V: y01->1, y11->(-1)
    #     W: z01->1, z11->1
    #
    #   Term 3: m4 = x11*(y10-y00)
    #     U: x11->1
    #     V: y10->1, y00->(-1)
    #     W: z10->1, z11->1
    #
    #   Term 4: m5 = (x00+x01)*y11
    #     U: x00->1, x01->1
    #     V: y11->1
    #     W: z00->(-1), z01->1
    #
    #   Term 5: m6 = (x10-x00)*(y00+y01)
    #     U: x10->1, x00->(-1)
    #     V: y00->1, y01->1
    #     W: z11->1
    #
    #   Term 6: m7 = (x01-x11)*(y10+y11)
    #     U: x01->1, x11->(-1)
    #     V: y10->1, y11->1
    #     W: z00->1
    #
    # Part 2: 16 columns (c = 28..43)
    #   For i in {0,1}, k in {0..7}:
    #     c = 28 + i*8 + k
    #     U[i*3 + 2, c] = 1  (A_1[i,0] where j=2)
    #     V[2*8 + k, c] = 1  (B_1[0,k] where j=2)
    #     W[k*2 + i, c] = 1  (C[i,k])

    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Part 1: 4 blocks of Strassen (28 columns)
    for b in range(4):
        for s in range(7):
            c = b * 7 + s
            # x indices: i in {0,1}, j in {0,1}
            # y indices: j in {0,1}, k_local in {0,1} -> k_global = 2*b + k_local
            # z indices: i in {0,1}, k_global in {2*b, 2*b+1}

            if s == 0:  # m1 = (x00+x11)*(y00+y11)
                U[0*3 + 0][c] = 1
                U[1*3 + 1][c] = 1
                V[0*8 + 2*b + 0][c] = 1
                V[1*8 + 2*b + 1][c] = 1
                W[(2*b + 0)*2 + 0][c] = 1
                W[(2*b + 1)*2 + 1][c] = 1
            elif s == 1:  # m2 = (x10+x11)*y00
                U[1*3 + 0][c] = 1
                U[1*3 + 1][c] = 1
                V[0*8 + 2*b + 0][c] = 1
                W[(2*b + 0)*2 + 1][c] = 1
                W[(2*b + 1)*2 + 1][c] = -1
            elif s == 2:  # m3 = x00*(y01-y11)
                U[0*3 + 0][c] = 1
                V[1*8 + 2*b + 1][c] = 1
                V[1*8 + 2*b + 0][c] = -1
                W[(2*b + 1)*2 + 0][c] = 1
                W[(2*b + 1)*2 + 1][c] = 1
            elif s == 3:  # m4 = x11*(y10-y00)
                U[1*3 + 1][c] = 1
                V[1*8 + 2*b + 0][c] = 1
                V[0*8 + 2*b + 0][c] = -1
                W[(2*b + 0)*2 + 1][c] = 1
                W[(2*b + 1)*2 + 1][c] = 1
            elif s == 4:  # m5 = (x00+x01)*y11
                U[0*3 + 0][c] = 1
                U[0*3 + 1][c] = 1
                V[1*8 + 2*b + 1][c] = 1
                W[(2*b + 0)*2 + 0][c] = -1
                W[(2*b + 1)*2 + 0][c] = 1
            elif s == 5:  # m6 = (x10-x00)*(y00+y01)
                U[1*3 + 0][c] = 1
                U[0*3 + 0][c] = -1
                V[0*8 + 2*b + 0][c] = 1
                V[1*8 + 2*b + 1][c] = 1
                W[(2*b + 1)*2 + 1][c] = 1
            elif s == 6:  # m7 = (x01-x11)*(y10+y11)
                U[0*3 + 1][c] = 1
                U[1*3 + 1][c] = -1
                V[1*8 + 2*b + 0][c] = 1
                V[1*8 + 2*b + 1][c] = 1
                W[(2*b + 0)*2 + 0][c] = 1

    # Part 2: 16 columns for A_1 * B_1
    for i in range(2):
        for k in range(8):
            c = 28 + i * 8 + k
            U[i*3 + 2][c] = 1
            V[2*8 + k][c] = 1
            W[k*2 + i][c] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
