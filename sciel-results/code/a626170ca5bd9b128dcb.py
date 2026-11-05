import numpy as np

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 6

    # We want to decompose the matrix multiplication tensor for 3x5 * 5x6 -> 3x6.
    # The tensor has indices (i, j, k) where i in [0,3), j in [0,5), k in [0,6).
    # U has shape (n*m, rank) = (15, rank)
    # V has shape (m*p, rank) = (30, rank)
    # W has shape (p*n, rank) = (18, rank)
    # Convention: W[k*n + i, r]

    # Strategy: Use Strassen-like decomposition.
    # R(3,5,6) can be bounded by padding to (4,8,8) or similar, but let's try to be smarter.
    
    # Known bounds:
    # R(3,3,3) = 23 (Bini et al.)
    # R(2,2,2) = 7 (Strassen)
    
    # Let's partition the 3x5x6 multiplication.
    # We can view A as 3x5, B as 5x6, C as 3x6.
    
    # One approach: Partition the middle dimension 5 into 2+3.
    # A = [A1 (3x2), A2 (3x3)]
    # B = [B1; B2] where B1 is 2x6, B2 is 3x6
    # C = A1*B1 + A2*B2
    
    # Rank <= R(3,2,6) + R(3,3,6)
    
    # Now bound R(3,2,6):
    # Partition 6 into 3+3.
    # R(3,2,6) <= R(3,2,3) + R(3,2,3)
    # R(3,2,3): Pad to 3x3x3? R(3,3,3)=23. So R(3,2,3) <= 23.
    # So R(3,2,6) <= 46.
    
    # Now bound R(3,3,6):
    # Partition 6 into 3+3.
    # R(3,3,6) <= R(3,3,3) + R(3,3,3) = 23 + 23 = 46.
    
    # Total rank <= 46 + 46 = 92.
    # This is slightly worse than schoolbook (90). Not good.
    
    # Let's try a different partition.
    # Partition the output 6 into 2+2+2.
    # R(3,5,6) <= 3 * R(3,5,2)
    # R(3,5,2): Pad 5 to 6? R(3,6,2).
    # R(3,6,2) <= 3 * R(3,2,2).
    # R(3,2,2) <= R(3,3,2) <= R(3,3,3) = 23? No, R(3,3,2) is smaller.
    # Actually R(2,2,2)=7. R(3,2,2) is not standard.
    
    # Let's use the known result for R(3,3,3) = 23.
    # Can we use this directly?
    # 3x5x6 can be embedded in 3x6x6.
    # R(3,6,6) <= 4 * R(3,3,3) = 4 * 23 = 92. Still > 90.
    
    # How about R(3,5,6) via 4x4x4 blocks?
    # Pad 3x5x6 to 4x8x8?
    # R(4,8,8) <= 2 * R(4,4,8) <= 4 * R(4,4,4).
    # R(4,4,4) = 14 (known optimal).
    # So R(4,8,8) <= 4 * 14 = 56.
    # This is much better than 90!
    
    # Let's verify the embedding.
    # We need to map (3,5,6) -> (4,8,8).
    # A (3x5) -> A' (4x8).
    # B (5x6) -> B' (8x8).
    # C (3x6) -> C' (4x8).
    
    # The multiplication is C = A B.
    # We construct A', B', C' such that C' = A' B' implies C = A B.
    
    # Let's define the embedding.
    # A' is 4x8. We place A in the top-left 3x5 block?
    # A' = [[A, 0], [0, 0]] where A is 3x5, so top-left is 3x5, top-right is 3x3, bottom-left is 1x5, bottom-right is 1x3.
    # Wait, 4x8.
    # Let's split the 8 columns into 5 and 3.
    # A' = [A | 0_{3x3}]
    #      [0_{1x5} | 0_{1x3}]
    
    # B' is 8x8.
    # We need A' B' to give C' which contains C.
    # C' is 4x8. We want C in the top-left 3x6 block?
    # C' = [C | 0_{3x2}]
    #      [0_{1x6} | 0_{1x2}]
    
    # Let's check:
    # A' B' = [A | 0] [B1 | B2]
    #         [0 | 0] [B3 | B4]
    #        = [A B1 | A B2]
    #          [0    | 0   ]
    
    # We want [A B1 | A B2] to be [C | 0].
    # So we need A B1 = C and A B2 = 0.
    # Since A is 3x5 and B1 is 5x6, A B1 = C is just the original multiplication.
    # We need to embed B (5x6) into B1 (5x6) part of B'.
    # But B' is 8x8.
    # B' = [B | 0_{5x2}]
    #      [0_{3x5} | 0_{3x2}]
    # Wait, B' must be 8x8.
    # If B' = [B | 0], where B is 5x6, then the top-left is 5x6.
    # The remaining parts of B' are 5x2 (top right), 3x5 (bottom left), 3x2 (bottom right).
    
    # Let's re-evaluate A'.
    # A' is 4x8.
    # If A' = [A | 0_{3x3}], then A' is 4x8.
    # Top 3 rows: [A (3x5) | 0 (3x3)]
    # Bottom 1 row: [0 (1x5) | 0 (1x3)]
    
    # Let B' be 8x8.
    # Top 5 rows: [B (5x6) | 0 (5x2)]
    # Bottom 3 rows: [0 (3x5) | 0 (3x2)]
    
    # Then A' B' =
    # Row 0-2 of A' times B':
    # [A | 0] * [B | 0; 0 | 0] = A B | 0.
    # Row 3 of A' is all zeros, so bottom row of C' is 0.
    
    # So C' = [C | 0; 0 | 0].
    # This works!
    
    # So the problem reduces to decomposing 4x8x8 multiplication.
    # R(4,8,8) <= 2 * R(4,4,8) ?
    # Let's split the 8 columns of A' into two 4-column blocks.
    # A' = [A1 | A2] where A1, A2 are 4x4.
    # B' = [B1; B2] where B1, B2 are 8x4? No.
    # B' is 8x8.
    # C' = A' B' = [A1 | A2] [B1; B2] = A1 B1 + A2 B2.
    # Here B1 is top 4 rows of B' (4x8), B2 is bottom 4 rows of B' (4x8).
    # Wait, B' is 8x8.
    # If we split B' vertically into two 8x4 matrices, say B_left and B_right.
    # And A' horizontally into two 4x4 matrices, A_left and A_right.
    # Then C' = A_left B_left + A_right B_right.
    # C' is 4x8.
    # B_left is 8x4, B_right is 8x4.
    # A_left is 4x4, A_right is 4x4.
    # So we need to compute two 4x4x4 multiplications?
    # No, A_left B_left is 4x4. A_right B_right is 4x4.
    # C' = [A_left B_left | A_right B_right]?
    # No, C' is 4x8.
    # If C' = [C1 | C2] where C1, C2 are 4x4.
    # Then C1 = A_left B_left and C2 = A_right B_right.
    # This requires decomposing two independent 4x4x4 multiplications.
    # Rank = 2 * R(4,4,4) = 2 * 14 = 28.
    
    # Wait, is B' structured such that B_left and B_right are independent?
    # B' = [B | 0; 0 | 0] (with B 5x6).
    # Let's look at the columns of B'.
    # B' has 8 columns.
    # Columns 0-5 contain B. Columns 6-7 are 0.
    # If we split B' into B_left (cols 0-3) and B_right (cols 4-7).
    # B_left = [B_{0:5, 0:4}; 0_{3x4}].
    # B_right = [B_{0:5, 4:6} | 0_{5x2}; 0_{3x4} | 0_{3x2}].
    # Wait, B is 5x6.
    # B_left is 8x4. Top 5 rows are B[:, 0:4]. Bottom 3 rows are 0.
    # B_right is 8x4. Top 5 rows are [B[:, 4:6], 0_{5x2}]. Bottom 3 rows are 0.
    
    # A' is 4x8.
    # A_left = A'[:, 0:4]. A_right = A'[:, 4:8].
    # A' = [A | 0; 0 | 0].
    # A is 3x5.
    # A_left = [A[:, 0:4] | 0_{3x0}? No.
    # A' columns 0-3: A[:, 0:4] (3x4) and 0 (1x4). So A_left is 4x4.
    # A' columns 4-7: [A[:, 4:5] | 0_{3x3}] (top 3 rows) and 0 (1x4). So A_right is 4x4.
    
    # So we have two 4x4x4 multiplications:
    # 1. A_left (4x4) * B_left (4x8? No, B_left is 8x4).
    # Wait, the multiplication is A' (4x8) * B' (8x8).
    # We split the inner dimension (8) into two 4s.
    # A' = [A1 | A2] (4x4, 4x4).
    # B' = [B1; B2] (8x4, 8x4).
    # C' = A1 B1 + A2 B2.
    # Here B1 is top 4 rows of B' (4x8), B2 is bottom 4 rows of B' (4x8).
    # Wait, B' is 8x8.
    # If we split B' horizontally (rows), B1 is 4x8, B2 is 4x8.
    # Then A1 (4x4) * B1 (4x8) = 4x8.
    # A2 (4x4) * B2 (4x8) = 4x8.
    # Sum is 4x8.
    # This is not two 4x4x4 multiplications. It is two 4x4x8 multiplications.
    # R(4,4,8) <= 2 * R(4,4,4) = 28.
    # So total rank <= 28 + 28 = 56.
    
    # This matches my earlier calculation.
    # So we need a rank-14 decomposition for 4x4x4.
    
    # Let's implement the rank-14 decomposition for 4x4x4.
    # This is a known result. I will try to find or construct it.
    # If I can't, I will fall back to Strassen 2x2x2 (rank 7) applied to 2x2 blocks?
    # 4x4x4 can be seen as 4 independent 2x2x2 multiplications? No.
    # But R(4,4,4) <= 2 * R(2,2,4)?
    # R(2,2,4) <= 2 * R(2,2,2) = 14.
    # So R(4,4,4) <= 2 * 14 = 28. Not 14.
    
    # The optimal R(4,4,4)=14 is non-trivial.
    # Given the complexity, I will use the schoolbook for 4x4x4 (rank 16) if I can't find 14?
    # 16 is close to 14.
    # 2 * 16 = 32.
    # Total rank 32.
    # This is much better than 90.
    
    # Let's use the schoolbook decomposition for the 4x4x4 sub-problems.
    # It is guaranteed to be correct.
    
    rank = 32
    U = [[0] * rank for _ in range(4 * 8)] # 32 rows
    V = [[0] * rank for _ in range(8 * 8)] # 64 rows
    W = [[0] * rank for _ in range(8 * 4)] # 32 rows
    
    # We need to map the 4x8x8 decomposition to the 3x5x6 problem.
    # The 4x8x8 problem is:
    # A' (4x8), B' (8x8), C' (4x8).
    # U' has shape (32, 32). V' has shape (64, 32). W' has shape (32, 32).
    
    # We will construct U, V, W for the 3x5x6 problem directly.
    # The indices for the 3x5x6 problem are:
    # U: (i, j) for i in 0..2, j in 0..4. (15 entries)
    # V: (j, k) for j in 0..4, k in 0..5. (30 entries)
    # W: (k, i) for k in 0..5, i in 0..2. (18 entries)
    
    # The indices for the 4x8x8 problem are:
    # U': (i', j') for i' in 0..3, j' in 0..7. (32 entries)
    # V': (j', k') for j' in 0..7, k' in 0..7. (64 entries)
    # W': (k', i') for k' in 0..7, i' in 0..3. (32 entries)
    
    # We need to map (i, j) -> (i', j'), (j, k) -> (j', k'), (k, i) -> (k', i').
    # From our embedding:
    # A (3x5) is in A' (4x8) at rows 0-2, cols 0-4.
    # So i' = i, j' = j.
    # B (5x6) is in B' (8x8) at rows 0-4, cols 0-5.
    # So j' = j, k' = k.
    # C (3x6) is in C' (4x8) at rows 0-2, cols 0-5.
    # So k' = k, i' = i.
    
    # So the mapping is identity for the non-zero parts.
    # We just need to extract the relevant parts of U', V', W'.
    
    # Let's construct U', V', W' for the 4x8x8 schoolbook (rank 64? No, we said 32 using two 4x4x8).
    # Wait, the 4x8x8 schoolbook is 4*8*8 = 256.
    # My "32 rank" estimate was for 2 * R(4,4,4) with R(4,4,4)=14? No, I used schoolbook for 4x4x4 (16).
    # So 2 * 16 = 32.
    # Let's verify the 4x4x8 decomposition.
    # 4x4x8 = two 4x4x4.
    # Rank 16 + 16 = 32.
    
    # So we will use 32 terms.
    # For each term t in 0..31:
    # If t < 16:
    #   It corresponds to the first 4x4x4 block.
    #   A1 (4x4) * B1 (4x8) -> C1 (4x8).
    #   Wait, the split was A' = [A1 | A2], B' = [B1; B2].
    #   C' = A1 B1 + A2 B2.
    #   A1 is 4x4 (cols 0-3 of A').
    #   B1 is 4x8 (rows 0-3 of B').
    #   A2 is 4x4 (cols 4-7 of A').
    #   B2 is 4x8 (rows 4-7 of B').
    
    # Let's refine the 4x4x8 decomposition.
    # A1 (4x4) * B1 (4x8).
    # This is 4x4x8.
    # We can split B1 into two 4x4 blocks: B1a (cols 0-3), B1b (cols 4-7).
    # C1 = A1 B1a + A1 B1b.
    # This is two 4x4x4 multiplications.
    # Rank 16 + 16 = 32.
    
    # So the total rank is 32.
    # The terms are:
    # 0-15: A1 * B1a
    # 16-31: A1 * B1b
    # Wait, I also have A2 * B2.
    # So it's 4 blocks of 4x4x4.
    # Rank 4 * 16 = 64.
    
    # Let's re-evaluate R(4,8,8).
    # R(4,8,8) <= 2 * R(4,4,8).
    # R(4,4,8) <= 2 * R(4,4,4).
    # R(4,4,4) = 16 (schoolbook).
    # So R(4,8,8) <= 4 * 16 = 64.
    
    # 64 is still better than 90.
    
    # Let's implement the 64-rank decomposition.
    rank = 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # We will iterate over the 4x8x8 schoolbook and map back to 3x5x6.
    # 4x8x8 schoolbook has 4*8*8 = 256 terms? No, we are using the block decomposition.
    # Let's just use the 3x5x6 schoolbook (90) if the mapping is complex.
    # 90 is the current best.
    # Can I get 64?
    
    # Let's try to map the 64 terms.
    # The 4x8x8 problem is decomposed into 4 4x4x4 problems.
    # Block 1: A1 (rows 0-3, cols 0-3) * B1a (rows 0-3, cols 0-3) -> C1a (rows 0-3, cols 0-3)
    # Block 2: A1 (rows 0-3, cols 0-3) * B1b (rows 0-3, cols 4-7) -> C1b (rows 0-3, cols 4-7)
    # Block 3: A2 (rows 0-3, cols 4-7) * B2a (rows 4-7, cols 0-3) -> C2a (rows 0-3, cols 0-3)
    # Block 4: A2 (rows 0-3, cols 4-7) * B2b (rows 4-7, cols 4-7) -> C2b (rows 0-3, cols 4-7)
    
    # Wait, C' = A1 B1 + A2 B2.
    # B1 is rows 0-3 of B'. B2 is rows 4-7 of B'.
    # B1 is 4x8. B2 is 4x8.
    # A1 is 4x4. A2 is 4x4.
    # C1 = A1 B1 is 4x8.
    # C2 = A2 B2 is 4x8.
    # C' = C1 + C2.
    
    # So we have two 4x4x8 multiplications.
    # Each 4x4x8 is two 4x4x4.
    # Total 4 4x4x4.
    
    # Let's map the indices.
    # For Block 1:
    # A1 indices: i' in 0..3, j' in 0..3.
    # B1a indices: j' in 0..3, k' in 0..3.
    # C1a indices: k' in 0..3, i' in 0..3.
    
    # Map to 3x5x6:
    # i' = i (0..2), j' = j (0..3).
    # So A1 has non-zero entries for i=0..2, j=0..3.
    # B1a has non-zero entries for j=0..3, k=0..3.
    # C1a has non-zero entries for k=0..3, i=0..2.
    
    # So for each term in Block 1:
    # U[i, j] += 1 if i<3, j<4.
    # V[j, k] += 1 if j<4, k<4.
    # W[k, i] += 1 if k<4, i<3.
    
    # We will add these to the global U, V, W.
    
    # Let's write a helper to add a 4x4x4 block.
    def add_block(u_offset, v_offset, w_offset, u_rows, u_cols, v_rows, v_cols, w_rows, w_cols):
        # u_rows, u_cols are the ranges in the 3x5x6 U matrix.
        # v_rows, v_cols are the ranges in the 3x5x6 V matrix.
        # w_rows, w_cols are the ranges in the 3x5x6 W matrix.
        pass
    
    # This is getting complicated. I will just use the schoolbook 90.
    # It is safe and correct.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
