"""Rewritten program: Strassen-based rank-56 decomposition for 2x4 * 4x8."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We view the 2x4 matrix A as two 2x2 blocks: A_top (rows 0,1 of the 4 cols)
    # Actually, let's be precise about the indexing.
    # A is 2x4. B is 4x8.
    # C is 2x8.
    #
    # Standard block decomposition:
    # A = [A11 A12]  (1x2 blocks of size 2x2? No, A is 2x4.
    # Let's split A into two 2x2 matrices: A_left (cols 0,1), A_right (cols 2,3)
    # B is 4x8. Split B into two 2x4 matrices? No, B is 4x8.
    # Let's split B vertically: B_top (rows 0,1), B_bottom (rows 2,3). Each is 2x8.
    #
    # C = A * B = A_left * B_top + A_right * B_bottom
    #
    # This is a sum of two matrix products: (2x2)*(2x8) + (2x2)*(2x8).
    # This doesn't directly help with Strassen on 2x2 unless we structure it differently.
    #
    # Alternative:
    # Split A into two 1x4 rows? No.
    #
    # Let's use the known result: Rank of 2x4 * 4x8 multiplication.
    # This is equivalent to multiplying a 2x4 matrix by a 4x8 matrix.
    #
    # Let's try to decompose it using Strassen on 2x2 blocks.
    # We can pad A to 4x4 and B to 4x4? No, dimensions are fixed.
    #
    # Consider the tensor for 2x4 * 4x8.
    # Indices: i in {0,1}, j in {0..3}, k in {0..7}.
    # T_{i,j,k} = sum_l A_{i,l} B_{l,k}.
    #
    # Let's group the j index. j=0,1 and j=2,3.
    # C_{i,k} = sum_{l=0}^1 A_{i,l} B_{l,k} + sum_{l=2}^3 A_{i,l} B_{l,k}
    #
    # Let A1 be the 2x2 submatrix of A with cols 0,1.
    # Let A2 be the 2x2 submatrix of A with cols 2,3.
    # Let B1 be the 2x8 submatrix of B with rows 0,1.
    # Let B2 be the 2x8 submatrix of B with rows 2,3.
    #
    # C = A1 * B1 + A2 * B2.
    #
    # Each product A1*B1 is (2x2)*(2x8).
    # This can be seen as two independent (2x2)*(2x4) multiplications? No.
    # A1*B1 computes 2 rows of output, each of length 8.
    #
    # Actually, (2x2)*(2x8) is equivalent to two (2x2)*(2x4) multiplications if we split B1 into two 2x4 blocks.
    # Let B1 = [B1a B1b] where B1a, B1b are 2x4.
    # Then A1*B1 = [A1*B1a, A1*B1b].
    # Each A1*B1a is (2x2)*(2x4).
    #
    # So the total problem is 4 independent (2x2)*(2x4) multiplications.
    # A1*B1a, A1*B1b, A2*B2a, A2*B2b.
    #
    # What is the rank of (2x2)*(2x4) multiplication?
    # This is multiplying a 2x2 matrix by a 2x4 matrix.
    # Output is 2x4.
    #
    # We can use Strassen for this?
    # Strassen is for 2x2 * 2x2.
    # For 2x2 * 2x4, we can split the 2x4 into two 2x2 matrices.
    # So it's two independent 2x2 * 2x2 multiplications.
    #
    # So the total problem is 8 independent 2x2 * 2x2 multiplications.
    #
    # If we use Strassen (rank 7) for each 2x2 * 2x2, the total rank is 8 * 7 = 56.
    #
    # Let's verify the decomposition.
    #
    # A is 2x4.
    # A = [A00 A01] where A00 is 2x2 (cols 0,1), A01 is 2x2 (cols 2,3).
    #
    # B is 4x8.
    # B = [B00 B01; B10 B11] where B00 is 2x4 (rows 0,1, cols 0,1,2,3)? No.
    # B is 4x8.
    # Rows 0,1 are the top half. Rows 2,3 are the bottom half.
    # Top half: B_top is 2x8.
    # Bottom half: B_bot is 2x8.
    #
    # C = A00 * B_top + A01 * B_bot.
    #
    # B_top is 2x8. Split into B_top_left (2x4) and B_top_right (2x4).
    # B_bot is 2x8. Split into B_bot_left (2x4) and B_bot_right (2x4).
    #
    # C = A00 * [B_tl B_tr] + A01 * [B_bl B_br]
    #   = [A00*B_tl + A01*B_bl, A00*B_tr + A01*B_br]
    #
    # So we have two independent 2x4 outputs: C_left and C_right.
    # C_left = A00*B_tl + A01*B_bl. (2x4 result)
    # C_right = A00*B_tr + A01*B_br. (2x4 result)
    #
    # Each of these is a sum of two (2x2)*(2x4) products.
    # A (2x2) * B (2x4) can be computed by splitting B into two 2x2 matrices.
    # Let B = [B1 B2] where B1, B2 are 2x2.
    # A*B = [A*B1, A*B2].
    #
    # So C_left = [A00*B_tl1 + A01*B_bl1, A00*B_tl2 + A01*B_bl2]
    # where B_tl1, B_tl2 are the two 2x2 halves of B_tl.
    #
    # So we have 4 independent (2x2)*(2x2) multiplications for C_left?
    # No, it's a sum.
    # C_left_col1 = A00*B_tl1 + A01*B_bl1.
    # This is not a single matrix multiplication, it's a sum of two.
    #
    # Wait, the tensor decomposition is linear.
    # We can just compute each (2x2)*(2x2) product separately and sum them.
    #
    # Let's list the 8 (2x2)*(2x2) products needed:
    # 1. A00 * B_tl1  (contributes to C[0:2, 0:2])
    # 2. A00 * B_tl2  (contributes to C[0:2, 2:4])
    # 3. A01 * B_bl1  (contributes to C[0:2, 0:2])
    # 4. A01 * B_bl2  (contributes to C[0:2, 2:4])
    # 5. A00 * B_tr1  (contributes to C[0:2, 4:6])
    # 6. A00 * B_tr2  (contributes to C[0:2, 6:8])
    # 7. A01 * B_br1  (contributes to C[0:2, 4:6])
    # 8. A01 * B_br2  (contributes to C[0:2, 6:8])
    #
    # Each of these 8 products can be computed with rank 7 using Strassen.
    # Total rank = 8 * 7 = 56.
    #
    # We need to construct U, V, W for the full 2x4 * 4x8 problem.
    #
    # U has shape (n*m, rank) = (8, 56).
    # V has shape (m*p, rank) = (32, 56).
    # W has shape (p*n, rank) = (16, 56).
    #
    # Indices:
    # U[i*m + j, c] where i in {0,1}, j in {0..3}.
    # V[j*p + k, c] where j in {0..3}, k in {0..7}.
    # W[k*n + i, c] where k in {0..7}, i in {0,1}.
    #
    # For each of the 8 Strassen blocks, we have 7 rank-1 terms.
    # Let's define the Strassen decomposition for 2x2 * 2x2.
    #
    # Standard Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A01-A10)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # We need to map this to the global indices.
    #
    # For block b (0..7):
    #   It corresponds to a specific A_sub (2x2) and B_sub (2x2).
    #   A_sub is a submatrix of A (2x4).
    #   B_sub is a submatrix of B (4x8).
    #
    # Let's define the mapping for each block.
    #
    # Block 0: A00 * B_tl1
    #   A rows: 0,1. A cols: 0,1.
    #   B rows: 0,1. B cols: 0,1.
    #   Output C rows: 0,1. C cols: 0,1.
    #
    # Block 1: A00 * B_tl2
    #   A rows: 0,1. A cols: 0,1.
    #   B rows: 0,1. B cols: 2,3.
    #   Output C rows: 0,1. C cols: 2,3.
    #
    # Block 2: A01 * B_bl1
    #   A rows: 0,1. A cols: 2,3.
    #   B rows: 2,3. B cols: 0,1.
    #   Output C rows: 0,1. C cols: 0,1.
    #
    # Block 3: A01 * B_bl2
    #   A rows: 0,1. A cols: 2,3.
    #   B rows: 2,3. B cols: 2,3.
    #   Output C rows: 0,1. C cols: 2,3.
    #
    # Block 4: A00 * B_tr1
    #   A rows: 0,1. A cols: 0,1.
    #   B rows: 0,1. B cols: 4,5.
    #   Output C rows: 0,1. C cols: 4,5.
    #
    # Block 5: A00 * B_tr2
    #   A rows: 0,1. A cols: 0,1.
    #   B rows: 0,1. B cols: 6,7.
    #   Output C rows: 0,1. C cols: 6,7.
    #
    # Block 6: A01 * B_br1
    #   A rows: 0,1. A cols: 2,3.
    #   B rows: 2,3. B cols: 4,5.
    #   Output C rows: 0,1. C cols: 4,5.
    #
    # Block 7: A01 * B_br2
    #   A rows: 0,1. A cols: 2,3.
    #   B rows: 2,3. B cols: 6,7.
    #   Output C rows: 0,1. C cols: 6,7.
    #
    # For each block, we have 7 Strassen multiplications.
    # Total 56 rank-1 terms.
    #
    # Let's construct U, V, W.
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    #
    # Strassen coefficients for A and B parts.
    # For each Strassen multiplication M_t (t=0..6):
    # We need to define which A elements and B elements are involved, and their signs.
    #
    # M0: (A00+A11)*(B00+B11)
    #   A: A00(+1), A11(+1)
    #   B: B00(+1), B11(+1)
    # M1: (A10+A11)*B00
    #   A: A10(+1), A11(+1)
    #   B: B00(+1)
    # M2: A00*(B01-B11)
    #   A: A00(+1)
    #   B: B01(+1), B11(-1)
    # M3: A11*(B10-B00)
    #   A: A11(+1)
    #   B: B10(+1), B00(-1)
    # M4: (A00+A10)*B11
    #   A: A00(+1), A10(+1)
    #   B: B11(+1)
    # M5: (A01-A10)*(B00+B10)
    #   A: A01(+1), A10(-1)
    #   B: B00(+1), B10(+1)
    # M6: (A01-A11)*(B01+B11)
    #   A: A01(+1), A11(-1)
    #   B: B01(+1), B11(+1)
    #
    # Let's define the Strassen terms as a list of tuples.
    # Each term is (a_coeffs, b_coeffs) where a_coeffs is a dict {local_idx: coeff}
    # and b_coeffs is a dict {local_idx: coeff}.
    #
    # Local indices for 2x2 matrix:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    #
    strassen_terms = [
        # M0
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        # M1
        ({2: 1, 3: 1}, {0: 1}),
        # M2
        ({0: 1}, {1: 1, 3: -1}),
        # M3
        ({3: 1}, {2: 1, 0: -1}),
        # M4
        ({0: 1, 2: 1}, {3: 1}),
        # M5
        ({1: 1, 2: -1}, {0: 1, 2: 1}),
        # M6
        ({1: 1, 3: -1}, {1: 1, 3: 1}),
    ]
    #
    # Output combination coefficients for C elements.
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    #
    # c_coeffs: dict {local_c_idx: list of (m_idx, coeff)}
    #
    c_coeffs = {
        0: [(0, 1), (3, 1), (4, -1), (6, 1)],   # C00
        1: [(2, 1), (4, 1)],                     # C01
        2: [(1, 1), (3, 1)],                     # C10
        3: [(0, 1), (1, -1), (2, 1), (5, 1)],   # C11
    }
    #
    # Now, for each of the 8 blocks, we assign 7 columns to U, V, W.
    #
    # Block definition:
    # block_id: (a_row_start, a_col_start, b_row_start, b_col_start, c_row_start, c_col_start)
    #
    # A is 2x4. A rows: 0,1.
    # B is 4x8. B rows: 0..3.
    # C is 2x8. C rows: 0,1.
    #
    blocks = [
        # (a_col_start, b_row_start, b_col_start, c_col_start)
        # A rows are always 0,1. C rows are always 0,1.
        (0, 0, 0, 0),   # Block 0: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        (0, 0, 2, 2),   # Block 1: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
        (2, 2, 0, 0),   # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
        (2, 2, 2, 2),   # Block 3: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
        (0, 0, 4, 4),   # Block 4: A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
        (0, 0, 6, 6),   # Block 5: A[0:2, 0:2] * B[0:2, 6:8] -> C[0:2, 6:8]
        (2, 2, 4, 4),   # Block 6: A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
        (2, 2, 6, 6),   # Block 7: A[0:2, 2:4] * B[2:4, 6:8] -> C[0:2, 6:8]
    ]
    #
    col = 0
    for (a_cs, b_rs, b_cs, c_cs) in blocks:
        for t in range(7):
            a_coeffs, b_coeffs = strassen_terms[t]
            # Fill U
            for local_a, coeff in a_coeffs.items():
                a_i = local_a // 2
                a_j = a_cs + (local_a % 2)
                U[a_i * m + a_j][col] += coeff
            # Fill V
            for local_b, coeff in b_coeffs.items():
                b_i = local_b // 2
                b_k = b_cs + (local_b % 2)
                # B row index is b_rs + b_i
                # B col index is b_k
                V[(b_rs + b_i) * p + b_k][col] += coeff
            # Fill W
            for local_c, c_terms in c_coeffs.items():
                for m_idx, coeff_c in c_terms:
                    if m_idx == t:
                        c_i = local_c // 2
                        c_k = c_cs + (local_c % 2)
                        # W index: k*n + i
                        W[c_k * n + c_i][col] += coeff_c
            col += 1
    #
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
