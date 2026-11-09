"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    # We'll use a block decomposition:
    # A (3x4) = [[A1 (2x2), A2 (2x2)], [A3 (1x2), A4 (1x2)]]
    # B (4x6) = [[B1 (2x3), B2 (2x3)], [B3 (2x3), B4 (2x3)]]
    # C (3x6) = [[C1 (2x3), C2 (2x3)], [C3 (1x3), C4 (1x3)]]
    #
    # C1 = A1*B1 + A2*B3
    # C2 = A1*B2 + A2*B4
    # C3 = A3*B1 + A4*B3
    # C4 = A3*B2 + A4*B4
    #
    # For the 2x2 * 2x3 products, we can use a Strassen-like method.
    # For 2x2 * 2x3, split the 2x3 into two 2x1 columns.
    # Actually, let's use a different approach: treat 2x2 * 2x3 as 
    # computing two 2x2 * 2x1 products.
    #
    # Instead, let's use the following: for 2x2 * 2x3, we can use
    # 7 multiplications by embedding into 2x2 * 2x2 with one zero column,
    # but that's not clean.
    #
    # Let me try a direct construction with rank < 72.
    # 
    # Alternative: use the fact that we can do 2x2 matrix mult with rank 7.
    # 
    # Let's try: pad A to 4x4 (add zero row), then use Strassen on 4x4*4x6.
    # 4x4 = 2x2 blocks of 2x2
    # 4x6 = 2x2 blocks of 2x3
    # 
    # Strassen for 2x2 blocks:
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
    # But our A21 = A22 = 0 (since we padded with zeros).
    # So M1 = A11*(B11+B22)
    # M2 = 0
    # M3 = A11*(B12-B22)
    # M4 = 0
    # M5 = (A11+A12)*B22
    # M6 = -A11*(B11+B12)
    # M7 = (A12-A11)*(B21+B22)
    #
    # C11 = M1 - M5 + M7 = A11*(B11+B22) - (A11+A12)*B22 + (A12-A11)*(B21+B22)
    #     = A11*B11 + A11*B22 - A11*B22 - A12*B22 + A12*B21 + A12*B22 - A11*B21 - A11*B22
    #     = A11*B11 - A11*B22 + A12*B21 - A11*B22  ... this is getting messy.
    #
    # Let me just implement the full Strassen and count the rank.
    
    # Use Strassen on padded 4x4 * 4x6
    # A_padded is 4x4, B is 4x6
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where A11,A12,A21,A22 are 2x2
    # B = [[B11, B12], [B21, B22]] where B11,B12,B21,B22 are 2x3
    
    # Strassen gives 7 multiplications of 2x2 * 2x3
    # Each 2x2 * 2x3 has schoolbook rank 2*2*3 = 12
    # Total rank = 7 * 12 = 84, which is worse than 72!
    
    # So Strassen on the padded version doesn't help directly.
    
    # Let me try a different block split.
    # 
    # What if I split the middle dimension (4) into 2+2, and use
    # a smarter decomposition for 3x2 * 2x6 and 3x2 * 2x6?
    #
    # A (3x4) = [A_left (3x2), A_right (3x2)]
    # B (4x6) = [[B_top (2x6)], [B_bottom (2x6)]]
    # C = A_left * B_top + A_right * B_bottom
    #
    # Each term is 3x2 * 2x6, which has schoolbook rank 3*2*6 = 36
    # Total = 72, no improvement.
    #
    # I need to find a decomposition with rank < 72.
    #
    # Let me try using the known result that 3x3x3 matrix multiplication
    # has rank 23 (or is it 19? Actually, the best known upper bound is 23).
    #
    # For 3x4x6, let me try to embed into a structure that uses
    # lower-rank sub-decompositions.
    #
    # Actually, let me just try to find a good decomposition by hand
    # or use a known algorithm.
    
    # Let me try: split A into 2x2 + 1x2 (rows), B into 2x3 + 2x3 (cols)
    # and use a tensor decomposition.
    
    # I'll use a construction based on the following idea:
    # The multiplication tensor for 3x4x6 can be decomposed using
    # the tensor product of smaller decompositions.
    
    # For now, let me try a simple improvement: use the fact that
    # some of the products are zero or can be shared.
    
    # Actually, let me just return the schoolbook for now and think more.
    # But the task asks for a better algorithm...
    
    # Let me try a concrete construction:
    # 
    # Split:
    # A (3x4) = [[a11, a12, a13, a14],
    #             [a21, a22, a23, a24],
    #             [a31, a32, a33, a34]]
    #
    # B (4x6) = [[b11, b12, b13, b14, b15, b16],
    #             [b21, b22, b23, b24, b25, b26],
    #             [b31, b32, b33, b34, b35, b36],
    #             [b41, b42, b43, b44, b45, b46]]
    #
    # C (3x6) = A * B
    #
    # I'll use a block decomposition with the following splits:
    # Rows of A: 2 + 1
    # Cols of A / Rows of B: 2 + 2
    # Cols of B: 3 + 3
    #
    # This gives 2x2 blocks in the middle dimension.
    
    n, m, p = 3, 4, 6
    
    # Define the block structure
    # A blocks: A11 (2x2), A12 (2x2), A21 (1x2), A22 (1x2)
    # B blocks: B11 (2x3), B12 (2x3), B21 (2x3), B22 (2x3)
    # C blocks: C11 (2x3), C12 (2x3), C21 (1x3), C22 (1x3)
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For 2x2 * 2x3, I can use a decomposition of rank 7 (like Strassen for 2x2x2, 
    # but adapted for 2x2x2x3).
    
    # Actually, the tensor for 2x2 * 2x3 multiplication:
    # This is the tensor T = sum_{i=1..2, j=1..2, k=1..3} e_i^A e_j^B e_k^C
    # where A is 2x2, B is 2x3, C is 2x3.
    # The schoolbook rank is 2*2*3 = 12.
    # Can we do better? 
    # 
    # I believe the rank of 2x2x2x3 matrix multiplication is 7 or 8.
    # Let me assume we can do it with rank 8 for now.
    
    # If each 2x2*2x3 takes rank 8, and we have 4 such products (but they share structure),
    # we might be able to do better.
    
    # Let me just implement the schoolbook for now and return it.
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
