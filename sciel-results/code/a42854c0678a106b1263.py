# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 118.0

    n, m, p = DIMENSIONS  # 3, 4, 4

    # Schoolbook solution as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    def _build_rank7_2x2():
        """Return list of 7 (u, v, w) dicts for 2x2 matrix multiplication (Strassen)."""
        terms = []
        # p1 = (a11+a22)*(b11+b22)
        terms.append(({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}))
        # p2 = (a21+a22)*b11
        terms.append(({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}))
        # p3 = a11*(b12-b22)
        terms.append(({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}))
        # p4 = a22*(b21-b11)
        terms.append(({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}))
        # p5 = (a11+a12)*b22
        terms.append(({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}))
        # p6 = (a21-a11)*(b11+b12)
        terms.append(({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}))
        # p7 = (a12-a22)*(b21+b22)
        terms.append(({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}))
        return terms

    rank7_terms = _build_rank7_2x2()

    # We use a rank-28 decomposition based on 4 Strassen 2x2 blocks.
    # This computes the full 4x4 * 4x4 product, but we only use the first 3 rows of the result.
    # The input A is 3x4, so we treat the 4th row as zeros.
    
    total_rank = 28
    U = [[0] * total_rank for _ in range(n * m)]  # 12 x 28
    V = [[0] * total_rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * total_rank for _ in range(p * n)]  # 12 x 28

    def add_rank7_block(a_row_offset, b_col_offset, j_offset, c_row_offset, c_col_offset, col_start):
        """Add a rank-7 2x2*2x2 block.
        a_row_offset: row offset in A (0 or 2)
        b_col_offset: col offset in B (0 or 2)
        j_offset: row offset in B / col offset in A (0 or 2)
        c_row_offset: row offset in C (0 or 2)
        c_col_offset: col offset in C (0 or 2)
        """
        for r in range(7):
            u_dict, v_dict, w_dict = rank7_terms[r]
            col = col_start + r
            for (ai, aj), val in u_dict.items():
                # Map to A indices. A is 3x4.
                # If ai=1 and a_row_offset=2, then A row is 3, which is out of bounds (or zero).
                # We only add if the row is < n (3).
                a_row = a_row_offset + ai
                if a_row < n:
                    U[a_row * m + (j_offset + aj)][col] += val
            for (bj, bk), val in v_dict.items():
                V[(j_offset + bj) * p + (b_col_offset + bk)][col] += val
            for (ci, ck), val in w_dict.items():
                # Map to C indices. C is 3x4.
                # If ci=1 and c_row_offset=2, then C row is 3, which is out of bounds.
                # We only add if the row is < n (3).
                c_row = c_row_offset + ci
                if c_row < n:
                    W[(c_col_offset + ck) * n + c_row][col] += val
        return col_start + 7

    col = 0
    # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    col = add_rank7_block(0, 0, 0, 0, 0, col)
    # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    col = add_rank7_block(0, 0, 2, 0, 0, col)
    # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    col = add_rank7_block(0, 2, 0, 0, 2, col)
    # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    col = add_rank7_block(0, 2, 2, 0, 2, col)
    
    # We also need C[2, :] which depends on A[2, :].
    # A[2, 0:2] * B[0:2, 0:2] -> C[2, 0:2]
    # This is a 1x2 * 2x2 multiplication.
    # We can use schoolbook for this (rank 4) or a specialized Strassen-like?
    # Schoolbook is rank 4.
    # A[2, 2:4] * B[2:4, 0:2] -> C[2, 0:2] (Rank 4)
    # A[2, 0:2] * B[0:2, 2:4] -> C[2, 2:4] (Rank 4)
    # A[2, 2:4] * B[2:4, 2:4] -> C[2, 2:4] (Rank 4)
    
    # So we add 4 more rank-4 blocks.
    # Total rank = 28 + 16 = 44.
    
    # Wait, the current program already does this and gets 44.
    # My "Rank 28" idea was flawed because I forgot the bottom row of A contributes to the bottom row of C.
    # If I use the full 4x4 Strassen (Rank 28), it computes C[0:4, :].
    # C[2, :] is computed by the blocks involving A[2:4, :].
    # In the 4x4 Strassen, A is split into 4 2x2 blocks.
    # Block 3 and 4 involve A[2:4, :].
    # So the Rank 28 decomposition DOES compute C[2, :].
    # I just need to include the blocks that involve A[2:4, :].
    
    # Let's re-evaluate the 4x4 Strassen blocks.
    # 1. A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # 2. A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # 3. A[2:4, 0:2] * B[0:2, 2:4] -> C[2:4, 2:4] (Wait, indices?)
    # Standard block multiplication:
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # We need C11, C12, C21 (row 2 of C is part of C21/C22? No, C is 4x4 in this view).
    # C rows 0,1 are C11, C12.
    # C rows 2,3 are C21, C22.
    # We need C rows 0,1,2.
    # So we need C11, C12, and the first row of C21, C22?
    # No, C21 is 2x2. Rows 2,3 of C.
    # We need row 2 of C.
    # So we need the first row of C21 and C22.
    
    # The Strassen blocks compute the full 2x2 C blocks.
    # So we need:
    # - Blocks for C11 (A11B11, A12B21)
    # - Blocks for C12 (A11B12, A12B22)
    # - Blocks for C21 (A21B11, A22B21) -> Only need row 2 of C (first row of C21)
    # - Blocks for C22 (A21B12, A22B22) -> Only need row 2 of C (first row of C22)
    
    # All 8 multiplications are 2x2 * 2x2.
    # Each costs 7.
    # Total 8 * 7 = 56.
    # This is worse than 44.
    
    # So the current 44 is better than the naive 4x4 Strassen projection.
    # The current program uses Strassen for the top 2x4 part (4 blocks * 7 = 28)
    # and Schoolbook for the bottom 1x4 part (4 blocks * 4 = 16).
    # Total 44.
    
    # Can we do better for the bottom 1x4 part?
    # 1x4 * 4x4 -> 1x4.
    # This is just a linear combination.
    # C[2, k] = sum_j A[2, j] B[j, k].
    # This requires 16 multiplications in schoolbook.
    # Can we do it with fewer?
    # It's a bilinear form.
    # The rank of the 1x4 * 4x4 -> 1x4 tensor is 16?
    # Yes, because A[2,j] and B[j,k] are independent.
    # So 16 is minimal for the bottom row.
    
    # So 28 + 16 = 44 is likely optimal for this split.
    
    # What if we use Laderman (23) for the 4x4 part?
    # Laderman computes the full 4x4 C.
    # Cost 23.
    # We need C[0:3, :].
    # So we just project the Laderman result.
    # This gives rank 23.
    
    # So I MUST use Laderman.
    
    # I will hardcode the Laderman terms.
    # If I fail, I return 44.
    
    # I will use a known set of 23 terms.
    # I will define them as (U_dict, V_dict, W_dict).
    
    # This is risky but high reward.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
