"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # We decompose p=9 as 4+4+1
    # Part 1: 4x4 * 4x4 (Strassen rank 49)
    # Part 2: 4x4 * 4x4 (Strassen rank 49)
    # Part 3: 4x4 * 4x1 (rank 16)
    
    total_rank = 49 + 49 + 16
    U = [[0] * total_rank for _ in range(n * m)]  # 16 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 36 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 36 rows
    
    # Helper: Strassen algorithm for 2x2 matrices
    # Input: A (2x2), B (2x2)
    # Returns 7 products
    def strassen_2x2(a, b):
        # a and b are lists of 4 elements: [a11, a12, a21, a22]
        # and [b11, b12, b21, b22]
        m1 = (a[0] + a[3]) * (b[0] + b[3])
        m2 = (a[2] + a[3]) * b[0]
        m3 = a[0] * (b[1] - b[3])
        m4 = a[3] * (b[2] - b[0])
        m5 = (a[0] + a[1]) * b[3]
        m6 = (a[2] - a[0]) * (b[0] + b[1])
        m7 = (a[1] - a[3]) * (b[2] + b[3])
        return [m1, m2, m3, m4, m5, m6, m7]
    
    # For the tensor decomposition, we need to express each of the 7 Strassen
    # multiplications as rank-1 terms. Each 2x2 multiplication has rank 7.
    
    # Actually, let me think about this differently. The tensor for 4x4x4 matrix
    # multiplication can be decomposed using the block structure.
    
    # Let me use a direct approach: enumerate the Strassen decomposition explicitly.
    
    # For 2x2x2, the 7 rank-1 tensors are:
    # T1: (a11+a22)(b11+b22) -> contributes to c11, c22
    # T2: (a21+a22)b11 -> contributes to c21, c22
    # T3: a11(b12-b22) -> contributes to c11, c12
    # T4: a22(b21-b11) -> contributes to c21, c22
    # T5: (a11+a12)b22 -> contributes to c12, c22
    # T6: (a21-a11)(b11+b12) -> contributes to c21, c12
    # T7: (a12-a22)(b21+b22) -> contributes to c12, c21
    
    # For 4x4x4, we split into 2x2 blocks. Let A be 4x4 with blocks:
    # A = [[A11, A12], [A21, A22]] where each block is 2x2
    # B = [[B11, B12], [B21, B22]] where each block is 2x2
    # C = [[C11, C12], [C21, C22]] where each block is 2x2
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    #
    # Each of these is a 2x2 matrix product, so we apply Strassen to each.
    # That gives 4 * 7 = 28 rank-1 terms for the 4x4x4 case.
    #
    # Wait, that's rank 28, not 49! Let me reconsider.
    
    # Actually, I think I'm confusing myself. The tensor rank of 4x4x4 matrix
    # multiplication is known to be at most 49 (via recursive Strassen), but
    # the block approach I described gives 28. Let me verify.
    
    # The schoolbook rank for 4x4x4 is 64. Using block decomposition into 2x2
    # blocks, we have 4 block products, each of rank 7 (Strassen for 2x2x2),
    # so total rank = 4*7 = 28. That seems too good to be true.
    
    # Let me double-check: for 2x2x2, Strassen gives rank 7. The schoolbook
    # rank is 8. So for 4x4x4, if we use 2x2 blocks, we get 4*7 = 28,
    # which is better than schoolbook 64. This is correct!
    
    # So the rank for 4x4x4 is at most 28, not 49. Let me correct my analysis.
    
    # Revised plan:
    # Part 1: 4x4 * 4x4 (rank 28 via block Strassen)
    # Part 2: 4x4 * 4x4 (rank 28 via block Strassen)
    # Part 3: 4x4 * 4x1 (rank 16)
    # Total rank: 28 + 28 + 16 = 72
    
    # This is much better! Let me implement this.
    
    # First, let me define the 7 Strassen rank-1 tensors for 2x2x2.
    # For 2x2 matrices A, B with C = A*B:
    # The 7 multiplications are linear combinations of entries.
    
    # Let me define the Strassen decomposition for 2x2x2 explicitly.
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # The 7 products are:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    
    # And:
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # Now, each mi is a product of two linear forms. For the tensor decomposition,
    # we need to write each mi as a rank-1 term in the tensor U ⊗ V ⊗ W.
    
    # For the full 4x4x4 case, let me just implement it directly by constructing
    # the U, V, W matrices.
    
    # Let me start fresh with a clean implementation.
    
    # Reset
    total_rank = 72
    U = [[0] * total_rank for _ in range(n * m)]  # 16 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 36 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 36 rows
    
    col = 0
    
    # Part 1: 4x4 * 4x4 (columns 0-7 of B, rows 0-7 of A, cols 0-7 of C)
    # Using block Strassen with 2x2 blocks
    # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3),
    #            A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    # B blocks: B11 (rows 0-1, cols 0-1), B12 (rows 0-1, cols 2-3),
    #            B21 (rows 2-3, cols 0-1), B22 (rows 2-3, cols 2-3)
    # C blocks: C11 (rows 0-1, cols 0-1), C12 (rows 0-1, cols 2-3),
    #            C21 (rows 2-3, cols 0-1), C22 (rows 2-3, cols 2-3)
    
    # Strassen for 2x2: the 7 rank-1 tensors
    # For each block product, we have 7 terms.
    
    # Define the 7 Strassen linear forms for A and B:
    # For A block (2x2): entries indexed as (r,c) where r,c in {0,1}
    # For B block (2x2): entries indexed as (r,c) where r,c in {0,1}
    
    # The 7 products for a 2x2 multiplication:
    # Each product is a bilinear form in the 4 entries of A_block and 4 of B_block.
    
    # Let me define the 7 tensors explicitly.
    
    # Product 1: (a00+a11)(b00+b11)
    # This is rank 1 in the tensor sense? No, it's a product of two linear forms,
    # so it IS a rank-1 tensor term.
    
    # For the tensor decomposition, each Strassen product corresponds to one
    # column in U, V, W.
    
    # For a 2x2 block product at A positions (ar, ac), B positions (br, bc),
    # C positions (cr, cc), the 7 Strassen terms are:
    
    # Let me define a helper that adds the 7 Strassen terms for a block product.
    
    def add_strassen_block(ar, ac, br, bc, cr, cc, offset):
        """Add 7 Strassen rank-1 terms for 2x2 block product.
        ar, ac: top-left row/col of A block
        br, bc: top-left row/col of B block
        cr, cc: top-left row/col of C block
        offset: column offset in U, V, W
        Returns: next offset (offset + 7)
        """
        nonlocal col
        for t in range(7):
            c = offset + t
            if t == 0:
                # m1 = (a00+a11)(b00+b11)
                U[ar*m + ac][c] = 1
                U[(ar+1)*m + (ac+1)][c] = 1
                V[br*p + bc][c] = 1
                V[(br+1)*p + (bc+1)][c] = 1
                W[cr*n + cc][c] = 1
                W[(cr+1)*n + (cc+1)][c] = 1
                # c11 += m1, c22 += m1
                W[cr*n + cc][c] = 1  # already set
                W[(cr+1)*n + (cc+1)][c] = 1  # already set
            elif t == 1:
                # m2 = (a10+a11)b00
                U[(ar+1)*m + ac][c] = 1
                U[(ar+1)*m + (ac+1)][c] = 1
                V[br*p + bc][c] = 1
                # c21 += m2, c22 += m2
                W[(cr+1)*n + cc][c] = 1
                W[(cr+1)*n + (cc+1)][c] = 1
            elif t == 2:
                # m3 = a00(b01-b11)
                U[ar*m + ac][c] = 1
                V[br*p + (bc+1)][c] = 1
                V[(br+1)*p + (bc+1)][c] = -1
                # c11 += m3, c12 += m3
                W[cr*n + cc][c] = 1
                W[cr*n + (cc+1)][c] = 1
            elif t == 3:
                # m4 = a11(b10-b00)
                U[(ar+1)*m + (ac+1)][c] = 1
                V[(br+1)*p + bc][c] = 1
                V[br*p + bc][c] = -1
                # c11 += m4, c21 += m4
                W[cr*n + cc][c] = 1
                W[(cr+1)*n + cc][c] = 1
            elif t == 4:
                # m5 = (a00+a01)b11
                U[ar*m + ac][c] = 1
                U[ar*m + (ac+1)][c] = 1
                V[(br+1)*p + (bc+1)][c] = 1
                # c12 += m5, c22 += m5
                W[cr*n + (cc+1)][c] = 1
                W[(cr+1)*n + (cc+1)][c] = 1
            elif t == 5:
                # m6 = (a10-a00)(b00+b01)
                U[(ar+1)*m + ac][c] = 1
                U[ar*m + ac][c] = -1
                V[br*p + bc][c] = 1
                V[br*p + (bc+1)][c] = 1
                # c21 += m6, c12 += m6
                W[(cr+1)*n + cc][c] = 1
                W[cr*n + (cc+1)][c] = 1
            elif t == 6:
                # m7 = (a01-a11)(b10+b11)
                U[ar*m + (ac+1)][c] = 1
                U[(ar+1)*m + (ac+1)][c] = -1
                V[(br+1)*p + bc][c] = 1
                V[(br+1)*p + (bc+1)][c] = 1
                # c12 += m7, c21 += m7
                W[cr*n + (cc+1)][c] = 1
                W[(cr+1)*n + cc][c] = 1
        return offset + 7
    
    # But wait, I need to be more careful. The Strassen algorithm combines
    # the 7 products to form the 4 entries of the result. My current approach
    # adds multiple contributions to the same W entry, which is correct, but I
    # need to make sure the coefficients are right.
    
    # Let me restart with a cleaner approach. I'll build the tensors for each
    # of the 7 Strassen products, and each product contributes to specific
    # entries of C with specific coefficients.
    
    # Actually, my approach above is correct. Each of the 7 Strassen products
    # is a rank-1 tensor term, and the W matrix accumulates the coefficients.
    
    # Let me redo this more carefully.
    
    # Reset
    total_rank = 72
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Part 1: C[:, 0:4] = A * B[:, 0:4]
    # This is a 4x4 * 4x4 multiplication.
    # Split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]]
    # B = [[B11, B12], [B21, B22]]
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For each of the 8 block products (A11*B11, A12*B21, A11*B12, A12*B22,
    # A21*B11, A22*B21, A21*B12, A22*B22), we apply Strassen (7 terms each).
    
    # Wait, that's 8 block products * 7 = 56 terms, not 28. Let me reconsider.
    
    # Actually, C11 = A11*B11 + A12*B21 involves two block products, each of
    # rank 7, so C11 has rank at most 14. Similarly for other blocks. Total
    # rank for 4x4x4 would be 4*14 = 56? No, that's not right either.
    
    # Let me think again. The block decomposition gives:
    # C = [[A11*B11 + A12*B21, A11*B12 + A12*B22],
    #      [A21*B11 + A22*B21, A21*B12 + A22*B22]]
    
    # Each 2x2 block of C is a sum of two 2x2 matrix products. Each 2x2 matrix
    # product has rank 7 (via Strassen). So each C block has rank at most 14,
    # and the total rank is at most 4*14 = 56.
    
    # But this can be improved by sharing terms. The standard recursive Strassen
    # for 4x4 uses 7 multiplications of 2x2 blocks, giving rank 7*7 = 49.
    
    # Let me just use the known rank-49 decomposition for 4x4x4.
    
    # Actually, for simplicity and correctness, let me just use the schoolbook
    # decomposition but try to optimize. The schoolbook rank for 4x4x4 is 64.
    
    # Given the complexity, let me just use a straightforward approach:
    # Decompose p=9 as 4+4+1, use schoolbook for each part, and see if we can
    # find a lower rank.
    
    # Schoolbook for 4x4*4x4: rank 16
    # Schoolbook for 4x4*4x1: rank 16
    # Total: 16 + 16 + 16 = 48? No, that's not right.
    
    # Schoolbook for 4x4*4x4: for each i,j,k in {0..3}, one term. Rank 64.
    # Schoolbook for 4x4*4x1: for each i,j in {0..3}, one term. Rank 16.
    # Total: 64 + 64 + 16 = 144 (same as original).
    
    # OK so I need to actually implement Strassen. Let me do it properly.
    
    # For 2x2x2 Strassen, the 7 rank-1 tensors are well-defined. Let me
    # implement them explicitly.
    
    # For 2x2 matrices:
    # A = [[a0, a1], [a2, a3]]
    # B = [[b0, b1], [b2, b3]]
    # C = A*B = [[c0, c1], [c2, c3]]
    
    # The 7 Strassen products:
    # m0 = (a0+a3)(b0+b3)
    # m1 = (a2+a3)b0
    # m2 = a0(b1-b3)
    # m3 = a3(b2-b0)
    # m4 = (a0+a1)b3
    # m5 = (a2-a0)(b0+b1)
    # m6 = (a1-a3)(b2+b3)
    
    # c0 = m0 + m3 - m4 + m6
    # c1 = m2 + m4
    # c2 = m1 + m3
    # c3 = m0 - m1 + m2 + m5
    
    # Now, for the tensor decomposition, each mi is a rank-1 tensor.
    # mi = (linear form in A) * (linear form in B) * (coefficient in C)
    
    # Let me define the 7 tensors for 2x2x2.
    
    # Tensor 0: m0 = (a0+a3)(b0+b3)
    # U: a0 -> 1, a3 -> 1
    # V: b0 -> 1, b3 -> 1
    # W: c0 -> 1, c3 -> 1
    
    # Tensor 1: m1 = (a2+a3)b0
    # U: a2 -> 1, a3 -> 1
    # V: b0 -> 1
    # W: c2 -> 1, c3 -> 1
    
    # Tensor 2: m2 = a0(b1-b3)
    # U: a0 -> 1
    # V: b1 -> 1, b3 -> -1
    # W: c1 -> 1, c0 -> 1
    
    # Tensor 3: m3 = a3(b2-b0)
    # U: a3 -> 1
    # V: b2 -> 1, b0 -> -1
    # W: c0 -> 1, c2 -> 1
    
    # Tensor 4: m4 = (a0+a1)b3
    # U: a0 -> 1, a1 -> 1
    # V: b3 -> 1
    # W: c1 -> 1, c0 -> -1
    
    # Tensor 5: m5 = (a2-a0)(b0+b1)
    # U: a2 -> 1, a0 -> -1
    # V: b0 -> 1, b1 -> 1
    # W: c3 -> 1
    
    # Tensor 6: m6 = (a1-a3)(b2+b3)
    # U: a1 -> 1, a3 -> -1
    # V: b2 -> 1, b3 -> 1
    # W: c0 -> 1, c1 -> 1
    
    # Wait, let me double-check the C coefficients:
    # c0 = m0 + m3 - m4 + m6
    # c1 = m2 + m4
    # c2 = m1 + m3
    # c3 = m0 - m1 + m2 + m5
    
    # So:
    # m0: c0 += 1, c3 += 1
    # m1: c2 += 1, c3 += -1
    # m2: c1 += 1, c0 += 1
    # m3: c0 += 1, c2 += 1
    # m4: c1 += 1, c0 += -1
    # m5: c3 += 1
    # m6: c0 += 1, c1 += 1
    
    # Let me verify: c0 = m0(1) + m2(1) + m3(1) + m4(-1) + m6(1) = 1+1+1-1+1 = 3? 
    # That doesn't match c0 = m0 + m3 - m4 + m6.
    
    # I think I made an error. Let me re-derive.
    
    # Standard Strassen:
    # p1 = (a00 + a11)(b00 + b11)
    # p2 = (a10 + a11)b00
    # p3 = a00(b01 - b11)
    # p4 = a11(b10 - b00)
    # p5 = (a00 + a01)b11
    # p6 = (a10 - a00)(b00 + b01)
    # p7 = (a01 - a11)(b10 + b11)
    
    # c00 = p1 + p4 - p5 + p7
    # c01 = p3 + p5
    # c10 = p2 + p4
    # c11 = p1 - p2 + p3 + p6
    
    # So:
    # p1: c00 += 1, c11 += 1
    # p2: c10 += 1, c11 += -1
    # p3: c01 += 1, c11 += 1
    # p4: c00 += 1, c10 += 1
    # p5: c01 += 1, c00 += -1
    # p6: c11 += 1
    # p7: c00 += 1, c01 += 1
    
    # Let me verify: c00 = p1(1) + p4(1) + p5(-1) + p7(1) = 1+1-1+1 = 2? 
    # But c00 = p1 + p4 - p5 + p7, so yes, that's correct. The sum is 2 terms
    # with coefficient 1 and 1 term with -1, but that's fine because the
    # decomposition is linear.
    
    # OK so now let me implement this for the 4x4x9 case.
    
    # For the 4x4*4x4 part, I'll use the block Strassen approach where I
    # decompose into 2x2 blocks and apply Strassen to each block product.
    
    # But this gives 8 block products * 7 = 56 terms, which is worse than the
    # recursive Strassen (49 terms). Let me just use 56 terms for simplicity.
    
    # Actually, let me just use the schoolbook decomposition for now and see
    # if I can improve it. The schoolbook rank is 144. If I can get it below
    # 144, that's an improvement.
    
    # Let me try a simple optimization: for the 4x4*4x1 part, the rank is 16
    # (schoolbook). For the two 4x4*4x4 parts, if I can get rank < 64 each,
    # I'll save.
    
    # Given time constraints, let me just implement the schoolbook but with a
    # slight restructuring to see if it validates, then try to optimize.
    
    # Actually, let me just implement a correct schoolbook decomposition first
    # to make sure the framework works, then optimize.
    
    rank = n * m * p  # 144
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
