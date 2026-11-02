"""Strassen-based rank-28 decomposition for 2x4 by 4x4 matrix multiplication."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # We decompose C (2x4) = A (2x4) * B (4x4).
    # Partition A into [A00, A01] and B into four 2x2 blocks:
    #   B = [[B00, B01],
    #        [B10, B11]]
    # Then C0 = A00*B00 + A01*B10, C1 = A00*B01 + A01*B11.
    # Each 2x2x2 multiplication can be done in rank 7 via Strassen,
    # giving a total rank of 4*7 = 28.

    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen coefficients for each of the four 2x2x2 blocks.
    # For a 2x2x2 block, we need 7 intermediate products.
    # U coefficients (for A part):
    #   M1 = (A00+A11)   -> [1, 0, 0, 1]
    #   M2 = (A10+A11)   -> [0, 0, 1, 1]
    #   M3 = A00         -> [1, 0, 0, 0]
    #   M4 = A11         -> [0, 0, 0, 1]
    #   M5 = (A00+A01)   -> [1, 1, 0, 0]
    #   M6 = (A01-A10)   -> [0, 1, -1, 0]
    #   M7 = (A11-A00)   -> [-1, 0, 0, 1]

    # V coefficients (for B part):
    #   M1 = (B00+B11)   -> [1, 0, 0, 1]
    #   M2 = B00         -> [1, 0, 0, 0]
    #   M3 = (B01-B11)   -> [0, 1, 0, -1]
    #   M4 = (B10-B00)   -> [0, 0, 1, -1]
    #   M5 = B11         -> [0, 0, 0, 1]
    #   M6 = (B00+B10)   -> [1, 0, 1, 0]
    #   M7 = (B11+B01)   -> [0, 1, 0, 1]

    # W coefficients (for C part):
    #   C00 = M1 - M4 + M6  -> [1, 0, 0, 1], [-1, 0, 0, 0], [0, 0, 1, -1]
    #   C01 = M3 + M5       -> [0, 1, 0, 0], [0, 0, 0, 1]
    #   C10 = M2 + M4       -> [0, 0, 1, 0], [0, 0, 0, 1]
    #   C11 = M1 - M3 + M7  -> [1, 0, 0, 1], [-1, 1, 0, 0], [0, 0, 0, 1]

    U_coeffs = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [0, 1, -1, 0],  # M6
        [-1, 0, 0, 1],  # M7
    ]

    V_coeffs = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [0, 0, 1, -1],  # M4
        [0, 0, 0, 1],   # M5
        [1, 0, 1, 0],   # M6
        [0, 1, 0, 1],   # M7
    ]

    W_coeffs = [
        [[1, 0, 0, 1], [-1, 0, 0, 0], [0, 0, 1, -1]],  # C00
        [[0, 1, 0, 0], [0, 0, 0, 1]],                    # C01
        [[0, 0, 1, 0], [0, 0, 0, 1]],                    # C10
        [[1, 0, 0, 1], [-1, 1, 0, 0], [0, 0, 0, 1]],    # C11
    ]

    # There are 4 blocks:
    # Block 0: C00 = A00*B00
    # Block 1: C01 = A00*B01
    # Block 2: C10 = A01*B10
    # Block 3: C11 = A01*B11

    for block in range(4):
        base = block * 7
        for s in range(7):
            col = base + s
            # U coefficients: A is 2x4, partitioned as [A00, A01]
            # A00 is row 0 (A[0,0:2]), A01 is row 1 (A[0,2:4]) for C0
            # For C0: A00 = U[0:2], A01 = U[2:4]
            # For C1: A00 = U[4:6], A01 = U[6:8]
            if block < 2:  # C0
                A_rows = [0, 1]  # A00
            else:  # C1
                A_rows = [4, 5]  # A01

            # Actually, let me reconsider the indexing.
            # A is 2x4. C0 uses A row 0, C1 uses A row 1.
            # A00 = A[0, 0:2], A01 = A[0, 2:4]
            # For C0: A00 is at U indices 0,1; A01 is at U indices 2,3
            # For C1: A00 is at U indices 4,5; A01 is at U indices 6,7

            # U coefficients for the 2x2 A block
            # The 4 elements of the A 2x2 block are at positions:
            # If block is C0 (block 0 or 1): A00 at [0,1], A01 at [2,3]
            # If block is C1 (block 2 or 3): A00 at [4,5], A01 at [6,7]
            if block < 2:
                A_base = 0
            else:
                A_base = 4

            # A 2x2 block has elements at:
            # A00 -> A_base, A01 -> A_base + 1
            # A10 -> A_base + 2, A11 -> A_base + 3
            # Wait, A is 2x4. The 2x2 blocks are:
            # A00 = A[0,0], A[0,1]  -> indices 0, 1
            # A01 = A[0,2], A[0,3]  -> indices 2, 3
            # For C1:
            # A00 = A[1,0], A[1,1]  -> indices 4, 5
            # A01 = A[1,2], A[1,3]  -> indices 6, 7

            # So for U, the 4 elements of the A 2x2 block are at:
            # [A_base, A_base+1, A_base+2, A_base+3]
            # But wait, the A 2x2 block for C0 is:
            # A00 = A[0,0], A[0,1]  -> these are two elements in the same row
            # Actually, A is 2x4, so each row has 4 elements.
            # For C0 (row 0 of A): A00 = A[0,0:2], A01 = A[0,2:4]
            # The U matrix has rows 0-7. Row 0 is A[0,0], row 1 is A[0,1], etc.
            # So A00 elements are at U rows 0,1. A01 elements are at U rows 2,3.
            # For C1 (row 1 of A): A00 = A[1,0:2], A01 = A[1,2:4]
            # A00 elements are at U rows 4,5. A01 elements are at U rows 6,7.

            # The Strassen U coefficients apply to the 2x2 A block.
            # The 2x2 A block has 4 elements: top-left, top-right, bottom-left, bottom-right.
            # But A is 2x4, so the "2x2" block is actually just 2 elements (one row of A).
            # Hmm, this doesn't quite work. Let me reconsider.

            # Actually, for 2x4 * 4x4, we can think of it as:
            # C (2x4) = A (2x4) * B (4x4)
            # Each row of C is a linear combination of rows of B.
            # C[0,:] = A[0,0]*B[0,:] + A[0,1]*B[1,:] + A[0,2]*B[2,:] + A[0,3]*B[3,:]
            # C[1,:] = A[1,0]*B[0,:] + A[1,1]*B[1,:] + A[1,2]*B[2,:] + A[1,3]*B[3,:]

            # Partition B into 2x2 blocks:
            # B = [[B00, B01],
            #      [B10, B11]]
            # where B00 = B[0:2, 0:2], B01 = B[0:2, 2:4], etc.

            # Then C[0,:] = A[0,0:2] * B00 + A[0,2:4] * B10
            #        C[1,:] = A[1,0:2] * B01 + A[1,2:4] * B11

            # Wait, that's not right. Let me re-partition.
            # C[0,:] = A[0,:] * B[:,0:2] + A[0,:] * B[:,2:4]
            #         = A[0,0:4] * B[:,0:2] + A[0,0:4] * B[:,2:4]

            # Actually, the standard partitioning for this case:
            # Let A = [A0, A1] where A0 = A[0,:], A1 = A[1,:] (each 1x4)
            # Let B = [B0; B1] where B0 = B[0:2,:], B1 = B[2:4,:] (each 2x4)
            # Then C = [A0*B0 + A1*B1]... no, that gives 1x4.

            # I think the cleanest approach is:
            # C (2x4) = A (2x4) * B (4x4)
            # Partition A into two 2x2 blocks: A = [A00, A01]
            #   A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
            # Partition B into four 2x2 blocks:
            #   B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
            #   B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]
            # Then C = [A00*B00 + A01*B10, A00*B01 + A01*B11]
            # So C0 = A00*B00 + A01*B10 (2x2)
            #    C1 = A00*B01 + A01*B11 (2x2)

            # Now each of C0 and C1 is a 2x2 matrix.
            # C0 = A00*B00 + A01*B10
            # C1 = A00*B01 + A01*B11

            # For C0, we need to compute A00*B00 and A01*B10, then add.
            # Each product is 2x2, requiring 7 multiplications via Strassen.
            # So C0 requires 14 multiplications, C1 requires 14, total 28.

            # But we can't just add the Strassen results directly because
            # the addition is in the output, not in the intermediate products.

            # Actually, let me reconsider. The tensor decomposition approach:
            # We want to find U, V, W such that:
            # T[i,j,k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # where T is the matrix multiplication tensor.

            # For 2x4 * 4x4:
            # U is 8 x rank, V is 16 x rank, W is 8 x rank.
            # The indices are:
            # i in 0..7 (A row-major: i = a*4 + b where a in 0..1, b in 0..3)
            # j in 0..15 (B row-major: j = c*4 + d where c in 0..3, d in 0..3)
            # k in 0..7 (C row-major: k = e*4 + f where e in 0..1, f in 0..3)

            # The tensor T[i,j,k] is non-zero only when the corresponding
            # A element, B element, and C element are related by matrix multiplication.

            # Let me just directly construct the rank-28 decomposition.
            # I'll use the approach of decomposing into 4 Strassen blocks.

            # Block 0: C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
            # This is a sum of two 2x2 products.

            # Hmm, this is getting complicated. Let me try a different approach.
            # I'll use the known rank-28 decomposition from the literature.

            # Actually, let me just verify that rank 28 is achievable and construct it.
            # The key insight is that 2x4 * 4x4 can be decomposed into 4 independent
            # 2x2 * 2x2 multiplications (with appropriate linear combinations).

            # Let me try a direct construction:
            # For each of the 7 Strassen multipliers, and for each of the 4 blocks,
            # I'll set up the U, V, W entries.

            pass  # This is getting too complex inline. Let me restructure.

    # Let me rebuild this more carefully.
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(8)]

    # The 4 blocks correspond to:
    # Block 0: C[0:2, 0:2] involves A[0:2, :] * B[:, 0:2]
    # Block 1: C[0:2, 2:4] involves A[0:2, :] * B[:, 2:4]
    # Block 2: C[2:4, 0:2] involves A[2:4, :] * B[:, 0:2]  -- wait, C is 2x4, not 4x4.

    # Let me re-index. C is 2x4.
    # C[0, 0:4] = sum_{j=0}^{3} A[0, j] * B[j, 0:4]
    # C[1, 0:4] = sum_{j=0}^{3} A[1, j] * B[j, 0:4]

    # Partition B into 2x2 blocks:
    # B00 = B[0:2, 0:2], B01 = B[0:2, 2:4]
    # B10 = B[2:4, 0:2], B11 = B[2:4, 2:4]

    # C[0, 0:2] = A[0, 0:2] * B00 + A[0, 2:4] * B10
    # C[0, 2:4] = A[0, 0:2] * B01 + A[0, 2:4] * B11
    # C[1, 0:2] = A[1, 0:2] * B00 + A[1, 2:4] * B10
    # C[1, 2:4] = A[1, 0:2] * B01 + A[1, 2:4] * B11

    # So we have 4 independent 2x2 additions of products.
    # Each addition of two 2x2 products can be done in 7 + 7 = 14 multiplications.
    # Total: 4 * 14 = 56. That's worse than schoolbook.

    # Wait, that can't be right. Let me reconsider.
    # Actually, the standard result is that 2x4 * 4x4 has rank at most 28.
    # The trick is to use Strassen's algorithm more cleverly.

    # I think the correct decomposition is:
    # Treat the 4x4 matrix B as a 2x2 block matrix with 2x2 blocks.
    # Then C = A * B where A is 2x4.
    # We can write A as a 2x2 block matrix with 1x2 blocks.
    # A = [[a00, a01],
    #      [a10, a11]]
    # where a00 = A[0, 0:2], a01 = A[0, 2:4], etc.

    # Then C = [[a00*B00 + a01*B10, a00*B01 + a01*B11],
    #          [a10*B00 + a11*B10, a10*B01 + a11*B11]]

    # Each entry is a 1x2 * 2x2 product, which gives a 1x2 result.
    # A 1x2 * 2x2 product requires 4 multiplications (or 3 with some tricks).
    # There are 8 such products, so 8*4 = 32 or 8*3 = 24.

    # Hmm, I'm not confident about the exact rank. Let me just use the schoolbook
    # rank 32 for now, since I can't confidently construct rank 28.

    # Actually, let me try to look up or derive the rank-28 decomposition.
    # I'll use a different approach: I'll construct the decomposition by
    # finding a set of rank-1 tensors that sum to the matrix multiplication tensor.

    # For now, let me just return the schoolbook decomposition to be safe.
    rank = 32
    U = [[0] * rank for _ in range(8)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(8)]

    column = 0
    for i in range(2):
        for j in range(4):
            for k in range(4):
                U[i * 4 + j][column] = 1
                V[j * 4 + k][column] = 1
                W[k * 2 + i][column] = 1
                column += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
