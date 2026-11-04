"""Matrix multiplication 2x3x8 with rank-minimal decomposition."""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # We decompose the 2x3x8 multiplication into 4 independent 2x3x2 sub-problems.
    # For each 2x3x2 block (A: 2x3, B: 3x2, C: 2x2), we use a rank-6 decomposition.
    # The 2x3 matrix A has entries:
    #   a00 a01 a02
    #   a10 a11 a12
    # The 3x2 matrix B has entries:
    #   b00 b01
    #   b10 b11
    #   b20 b21
    # The 2x2 result C has entries:
    #   c00 = a00*b00 + a01*b10 + a02*b20
    #   c01 = a00*b01 + a01*b11 + a02*b21
    #   c10 = a10*b00 + a11*b10 + a12*b20
    #   c11 = a10*b01 + a11*b11 + a12*b21
    
    # We can view this as:
    #   [a00 a01] * [b00 b01] + a02 * [b20 b21] for row 0
    #   [a10 a11] * [b10 b11] + a12 * [b20 b21] for row 1
    # Wait, that's not quite right. Let me re-examine.
    #
    # c00 = a00*b00 + a01*b10 + a02*b20
    # c01 = a00*b01 + a01*b11 + a02*b21
    # c10 = a10*b00 + a11*b10 + a12*b20
    # c11 = a10*b01 + a11*b11 + a12*b21
    #
    # Let's group the first two terms:
    # Row 0 of C: [a00, a01] * [[b00, b01], [b10, b11]] + a02 * [b20, b21]
    # Row 1 of C: [a10, a11] * [[b00, b01], [b10, b11]] + a12 * [b20, b21]
    #
    # So C = [[a00, a01], [a10, a11]] * [[b00, b01], [b10, b11]] + 
    #        [[a02], [a12]] * [[b20, b21]]
    #
    # The first part is a 2x2 matrix multiplication (rank 7 via Strassen).
    # The second part is a 2x1 times 1x2 (outer product), which is rank 2 (or we can do it in 2 mults).
    # Wait, the outer product of a 2x1 and 1x2 is a 2x2 matrix with 4 entries, but it only needs 2 multiplications if we factor it properly? No, it needs 2 multiplications: a02*b20, a02*b21, a12*b20, a12*b21. That's 4 multiplications.
    # But wait, we can write it as:
    # [a02, a12]^T * [b20, b21] = 
    #   a02*b20  a02*b21
    #   a12*b20  a12*b21
    # This requires 4 scalar multiplications.
    #
    # So total for one 2x3x2 block: 7 (Strassen) + 4 (outer product) = 11? 
    # But the schoolbook is 12. Can we do better?
    #
    # Actually, let's look at the structure more carefully.
    # We have 6 inputs from A (a00,a01,a02,a10,a11,a12) and 4 inputs from B (b00,b01,b10,b11,b20,b21) - wait, B is 3x2 so it has 6 inputs too.
    # Total inputs: 12. Total outputs: 4.
    #
    # Let's try to find a rank-6 decomposition for 2x3x2.
    # Actually, it's known that M(2,3,2) = 12? Or can it be less?
    # Let's check: M(2,3,2) >= M(2,2,2) = 7. Also M(2,3,2) >= 2*3 = 6 (trivial).
    # I suspect the minimum is 12, but let's try to construct a rank-10 or rank-11.
    #
    # Alternative: use the fact that we have 4 blocks. Maybe we can share some multiplications across blocks?
    # The A matrix is the same for all 4 blocks. The B matrix is different for each block.
    # So we can precompute some combinations of A's entries once, and use them across all 4 blocks.
    #
    # Let's define the Strassen variables for the 2x2 part:
    # For each block k (k=0,1,2,3), the 2x2 part uses:
    #   A_k = [[a00, a01], [a10, a11]] (same for all k!)
    #   B_k = [[b00^(k), b01^(k)], [b10^(k), b11^(k)]]
    #
    # Strassen for A_k * B_k:
    #   M1 = (a00+a10) * (b00^(k)+b10^(k))
    #   M2 = (a10+a11) * (b01^(k)+b11^(k))  -- wait, standard Strassen is:
    #   M1 = (A11+A22)(B11+B22)
    #   M2 = (A21+A22)B11
    #   M3 = A11(B21-B22)
    #   M4 = A22(B11-B21)
    #   M5 = (A11+A12)B22
    #   M6 = (A21-A11)(B11+B12)
    #   M7 = (A12-A22)(B21+B22)
    #
    # Where A11=a00, A12=a01, A21=a10, A22=a11
    #   B11=b00^(k), B12=b01^(k), B21=b10^(k), B22=b11^(k)
    #
    # So:
    #   M1 = (a00+a11) * (b00^(k)+b11^(k))
    #   M2 = (a10+a11) * b00^(k)
    #   M3 = a00 * (b10^(k)-b11^(k))
    #   M4 = a11 * (b00^(k)-b10^(k))
    #   M5 = (a00+a01) * b11^(k)
    #   M6 = (a10-a00) * (b00^(k)+b01^(k))
    #   M7 = (a01-a11) * (b10^(k)+b11^(k))
    #
    # The A-side combinations are:
    #   S1 = a00+a11
    #   S2 = a10+a11
    #   S3 = a00
    #   S4 = a11
    #   S5 = a00+a01
    #   S6 = a10-a00
    #   S7 = a01-a11
    #
    # These 7 combinations can be computed once (they don't depend on k).
    # But in the tensor decomposition, each multiplication is a separate rank-1 term.
    # We can't "share" the computation of S1 across different rank-1 terms in the standard decomposition.
    # Each rank-1 term is u * v * w.
    #
    # So for each block k, we have 7 Strassen terms + 4 outer product terms = 11 terms.
    # Total: 4 * 11 = 44.
    #
    # Can we do better? What if we use a different decomposition for 2x3x2?
    # Let's try to find a rank-6 decomposition for 2x3x2.
    #
    # Actually, I recall that for 2x3x2, the minimum rank is 12. But let's try to construct a rank-10.
    #
    # Alternative approach: use the fact that the 2x3x8 can be seen as 2x3x8 = 2x3x(2*4).
    # We can use a block Strassen-like approach.
    #
    # Let's just go with the 44-rank decomposition for now, which is better than 48.
    #
    # Actually, wait. Let me reconsider the outer product part.
    # The outer product is:
    #   [a02, a12]^T * [b20^(k), b21^(k)]
    # This gives:
    #   c00 += a02*b20^(k)
    #   c01 += a02*b21^(k)
    #   c10 += a12*b20^(k)
    #   c11 += a12*b21^(k)
    #
    # We can write this as 4 rank-1 terms:
    #   T1: a02 * b20^(k) -> c00
    #   T2: a02 * b21^(k) -> c01
    #   T3: a12 * b20^(k) -> c10
    #   T4: a12 * b21^(k) -> c11
    #
    # So total per block: 7 + 4 = 11. Total: 44.
    #
    # Let's implement this.
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44
    
    col = 0
    for k in range(4):
        # Block k: B columns 2k and 2k+1
        # Strassen terms for 2x2 part
        # M1 = (a00+a11) * (b00^(k)+b11^(k))
        U[0][col] += 1  # a00
        U[3][col] += 1  # a11
        V[2*k][col] += 1  # b00^(k) = B[0, 2k]
        V[8+2*k+1][col] += 1  # b11^(k) = B[1, 2k+1]
        # c00 += M1, c11 += M1
        W[4*k][col] += 1  # C[0, 2k]
        W[4*k+3][col] += 1  # C[1, 2k+1]
        col += 1
        
        # M2 = (a10+a11) * b00^(k)
        U[3][col] += 1  # a10
        U[4][col] += 1  # a11
        V[2*k][col] += 1  # b00^(k)
        # c10 += M2, c11 += M2
        W[4*k+2][col] += 1  # C[1, 2k]
        W[4*k+3][col] += 1  # C[1, 2k+1]
        col += 1
        
        # M3 = a00 * (b10^(k)-b11^(k))
        U[0][col] += 1  # a00
        V[8+2*k][col] += 1  # b10^(k) = B[1, 2k]
        V[8+2*k+1][col] -= 1  # b11^(k) = B[1, 2k+1]
        # c01 += M3, c11 -= M3
        W[4*k+1][col] += 1  # C[0, 2k+1]
        W[4*k+3][col] -= 1  # C[1, 2k+1]
        col += 1
        
        # M4 = a11 * (b00^(k)-b10^(k))
        U[4][col] += 1  # a11
        V[2*k][col] += 1  # b00^(k)
        V[8+2*k][col] -= 1  # b10^(k)
        # c01 += M4, c11 -= M4
        W[4*k+1][col] += 1  # C[0, 2k+1]
        W[4*k+3][col] -= 1  # C[1, 2k+1]
        col += 1
        
        # M5 = (a00+a01) * b11^(k)
        U[0][col] += 1  # a00
        U[1][col] += 1  # a01
        V[8+2*k+1][col] += 1  # b11^(k)
        # c00 += M5, c11 += M5
        W[4*k][col] += 1  # C[0, 2k]
        W[4*k+3][col] += 1  # C[1, 2k+1]
        col += 1
        
        # M6 = (a10-a00) * (b00^(k)+b01^(k))
        U[3][col] += 1  # a10
        U[0][col] -= 1  # a00
        V[2*k][col] += 1  # b00^(k)
        V[2*k+1][col] += 1  # b01^(k) = B[0, 2k+1]
        # c10 += M6, c11 += M6
        W[4*k+2][col] += 1  # C[1, 2k]
        W[4*k+3][col] += 1  # C[1, 2k+1]
        col += 1
        
        # M7 = (a01-a11) * (b10^(k)+b11^(k))
        U[1][col] += 1  # a01
        U[4][col] -= 1  # a11
        V[8+2*k][col] += 1  # b10^(k)
        V[8+2*k+1][col] += 1  # b11^(k)
        # c01 += M7, c11 += M7
        W[4*k+1][col] += 1  # C[0, 2k+1]
        W[4*k+3][col] += 1  # C[1, 2k+1]
        col += 1
        
        # Outer product terms
        # T1: a02 * b20^(k) -> c00
        U[2][col] += 1  # a02
        V[16+2*k][col] += 1  # b20^(k) = B[2, 2k]
        W[4*k][col] += 1  # C[0, 2k]
        col += 1
        
        # T2: a02 * b21^(k) -> c01
        U[2][col] += 1  # a02
        V[16+2*k+1][col] += 1  # b21^(k) = B[2, 2k+1]
        W[4*k+1][col] += 1  # C[0, 2k+1]
        col += 1
        
        # T3: a12 * b20^(k) -> c10
        U[5][col] += 1  # a12
        V[16+2*k][col] += 1  # b20^(k)
        W[4*k+2][col] += 1  # C[1, 2k]
        col += 1
        
        # T4: a12 * b21^(k) -> c11
        U[5][col] += 1  # a12
        V[16+2*k+1][col] += 1  # b21^(k)
        W[4*k+3][col] += 1  # C[1, 2k+1]
        col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
