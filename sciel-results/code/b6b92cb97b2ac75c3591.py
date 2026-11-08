"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Known best rank for 4x4x4 is 49 (using Strassen 2x2x2 with rank 7)
    # We will construct the 49-rank decomposition using block-structured approach
    # where each block follows the 2x2 Strassen pattern.
    
    rank = 49
    
    # For 4x4x4, we can use the tensor product approach:
    # If we have a rank-r decomposition for 2x2x2, we can build 4x4x4 with rank r^2
    # The 2x2 Strassen has rank 7, so 4x4x4 would have rank 7^2 = 49.
    
    # Let's define the Strassen 2x2x2 decomposition first.
    # For 2x2 matrices A, B, C where C = A*B:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # For 4x4, we partition into 2x2 blocks of 2x2 submatrices.
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    # C = [[C11, C12], [C21, C22]] where each Cij is 2x2
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # Using Strassen on each 2x2 block multiplication, we need:
    # For C11: 7 multiplications for A11*B11, 7 for A12*B21 -> but we can share
    # Actually, the total rank is 49 = 7*7, coming from Kronecker product.
    
    # Let's build it directly using the Kronecker product structure.
    # Index in 4x4 matrix: row index i in [0,3], col index j in [0,3]
    # Block indices: bi = i//2, bj = j//2 (block position)
    # Within-block indices: wi = i%2, wj = j%2 (position within 2x2 block)
    
    # U has shape (16, 49): row indexed by (i, j) for A[i,j]
    # V has shape (16, 49): row indexed by (j, k) for B[j,k]
    # W has shape (16, 49): row indexed by (k, i) for C[k,i] (per task: W[k*n+i,r])
    
    # The 7 Strassen terms for 2x2:
    # Term 0: (a11+a22)*(b11+b22)
    # Term 1: (a21+a22)*b11
    # Term 2: a11*(b12-b22)
    # Term 3: a22*(b21-b11)
    # Term 4: (a11+a12)*b22
    # Term 5: (a21-a11)*(b11+b12)
    # Term 6: (a12-a22)*(b21+b22)
    
    # For 4x4, we have 2x2 blocks, so we need to combine these.
    # The rank-49 decomposition comes from the fact that:
    # rank(M_{4,4,4}) <= rank(M_{2,2,2})^2 = 49
    
    # Let's construct U, V, W with 49 columns.
    # Column index r in [0,48] corresponds to (r1, r2) where r1 = r//7, r2 = r%7
    # r1 indexes which block-level Strassen term, r2 indexes within-block Strassen term.
    
    # Actually, let me think more carefully. The Kronecker product approach:
    # If T_{2,2,2} = sum_{r=0}^{6} u_r ⊗ v_r ⊗ w_r
    # Then T_{4,4,4} can be written using the block structure.
    
    # Let me just build it explicitly.
    
    # For each block pair (bi, bj) in A and (bj', bk) in B contributing to (bi, bk) in C:
    # C[bi,bk] = sum_{bj} A[bi,bj] * B[bj,bk]
    
    # Each A[bi,bj] is a 2x2 matrix with elements A[2*bi+wi, 2*bj+wj]
    # Similarly for B and C.
    
    # The 49-rank decomposition:
    # For each block-level Strassen term (s1, s2) in [0,6]^2, and for each 
    # within-block Strassen term s3 in [0,6], we have one rank-1 term.
    # Total: 7*7 = 49 terms.
    
    # Wait, that's not quite right either. Let me think again.
    
    # Actually, the standard result is that rank(M_{4,4,4}) = 49, achieved by:
    # Using the 2x2 Strassen recursively. The 4x4 matrix multiplication can be 
    # done with 49 scalar multiplications.
    
    # Let me construct it using the following approach:
    # Define the 2x2 Strassen coefficients.
    
    # For 2x2: A = [[a1, a2], [a3, a4]], B = [[b1, b2], [b3, b4]]
    # C = [[c1, c2], [c3, c4]]
    # 
    # m1 = (a1+a4)(b1+b4)
    # m2 = (a3+a4)b1
    # m3 = a1(b2-b4)
    # m4 = a4(b3-b1)
    # m5 = (a1+a2)b4
    # m6 = (a3-a1)(b1+b2)
    # m7 = (a2-a4)(b3+b4)
    #
    # c1 = m1 + m4 - m5 - m7
    # c2 = m3 + m5
    # c3 = m2 + m4
    # c4 = m1 - m2 + m3 + m7
    
    # For 4x4, partition into 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where Aij is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # To compute C00, C01, C10, C11 using Strassen at the block level:
    # We'd need to compute 7 block multiplications, each of which is a 2x2*2x2.
    # But each 2x2*2x2 needs 7 scalar multiplications (Strassen).
    # So total = 7 * 7 = 49.
    
    # Let me define the block-level Strassen:
    # Let A' = [[A00, A01], [A10, A11]], B' = [[B00, B01], [B10, B11]]
    # These are 2x2 "matrices" over 2x2 matrices.
    
    # Block-level Strassen terms:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # Each Mj is a 2x2 matrix product, which itself can be done with 7 scalar mults.
    # So we have 7 block terms, each with 7 scalar terms = 49 total.
    
    # Now, for each block-level term Mj, we need to express it as a sum of
    # rank-1 terms at the scalar level.
    
    # For a generic 2x2*2x2 = 2x2 result, the Strassen gives 7 terms.
    # For each block-level term Mj, we apply the 2x2 Strassen to get 7 scalar terms.
    
    # Let's define the 2x2 Strassen in terms of coefficients.
    # For matrices P = [[p1, p2], [p3, p4]], Q = [[q1, q2], [q3, q4]]:
    # s1 = (p1+p4)(q1+q4)
    # s2 = (p3+p4)q1
    # s3 = p1(q2-q4)
    # s4 = p4(q3-q1)
    # s5 = (p1+p2)q4
    # s6 = (p3-p1)(q1+q2)
    # s7 = (p2-p4)(q3+q4)
    #
    # R11 = s1 + s4 - s5 - s7
    # R12 = s3 + s5
    # R21 = s2 + s4
    # R22 = s1 - s2 + s3 + s7
    
    # Now, for each block-level term Mj, we need to express the contributing
    # A and B submatrices.
    
    # Let me define helper functions.
    
    # For each of the 7 block-level terms, and for each of the 7 scalar-level terms,
    # we have a rank-1 term in the overall 4x4 decomposition.
    
    # Column index: r = j*7 + s, where j is block-level term (0-6), s is scalar-level term (0-6)
    
    # For column r, we need to determine:
    # - Which elements of A are involved and with what coefficients -> U[row, r]
    # - Which elements of B are involved and with what coefficients -> V[row, r]
    # - Which elements of C are involved and with what coefficients -> W[row, r]
    
    # Let's define the structure.
    
    # First, define the 2x2 Strassen coefficients for P and Q.
    # For scalar term s (0-6), P coefficient vector (for p1, p2, p3, p4) and Q coefficient vector (for q1, q2, q3, q4).
    
    # s=0: P=[1,0,0,1], Q=[1,0,0,1]
    # s=1: P=[0,0,1,1], Q=[1,0,0,0]
    # s=2: P=[1,0,0,0], Q=[0,1,0,-1]
    # s=3: P=[0,0,0,1], Q=[-1,0,1,0]
    # s=4: P=[1,1,0,0], Q=[0,0,0,1]
    # s=5: P=[-1,0,1,0], Q=[1,1,0,0]
    # s=6: P=[0,1,0,-1], Q=[0,0,1,1]
    
    P_coeff = [
        [1,0,0,1],
        [0,0,1,1],
        [1,0,0,0],
        [0,0,0,1],
        [1,1,0,0],
        [-1,0,1,0],
        [0,1,0,-1]
    ]
    Q_coeff = [
        [1,0,0,1],
        [1,0,0,0],
        [0,1,0,-1],
        [-1,0,1,0],
        [0,0,0,1],
        [1,1,0,0],
        [0,0,1,1]
    ]
    
    # C coefficient vectors for each scalar term s, giving coefficients for (R11, R12, R21, R22)
    # s=0: [1,0,0,1]
    # s=1: [0,0,1,0]
    # s=2: [0,1,0,1]
    # s=3: [1,0,0,0]
    # s=4: [-1,1,0,0]
    # s=5: [0,0,1,0]
    # s=6: [-1,0,0,1]
    
    C_coeff = [
        [1,0,0,1],
        [0,0,1,0],
        [0,1,0,1],
        [1,0,0,0],
        [-1,1,0,0],
        [0,0,1,0],
        [-1,0,0,1]
    ]
    
    # Now, for each block-level term j (0-6), we need to define:
    # - How A contributes: which 2x2 blocks of A, with what coefficients
    # - How B contributes: which 2x2 blocks of B, with what coefficients
    # - How C is affected: which 2x2 blocks of C, with what coefficients
    
    # Block-level Strassen:
    # j=0: M1 = (A00+A11)*(B00+B11)
    #   A: A00 with coeff 1, A11 with coeff 1
    #   B: B00 with coeff 1, B11 with coeff 1
    #   C: C00 coeff 1, C11 coeff 1
    
    # j=1: M2 = (A10+A11)*B00
    #   A: A10 coeff 1, A11 coeff 1
    #   B: B00 coeff 1
    #   C: C10 coeff 1
    
    # j=2: M3 = A00*(B01-B11)
    #   A: A00 coeff 1
    #   B: B01 coeff 1, B11 coeff -1
    #   C: C01 coeff 1
    
    # j=3: M4 = A11*(B10-B00)
    #   A: A11 coeff 1
    #   B: B10 coeff 1, B00 coeff -1
    #   C: C00 coeff 1, C10 coeff 1
    
    # j=4: M5 = (A00+A01)*B11
    #   A: A00 coeff 1, A01 coeff 1
    #   B: B11 coeff 1
    #   C: C00 coeff -1, C01 coeff 1
    
    # j=5: M6 = (A10-A00)*(B00+B01)
    #   A: A10 coeff 1, A00 coeff -1
    #   B: B00 coeff 1, B01 coeff 1
    #   C: C11 coeff 1
    
    # j=6: M7 = (A01-A11)*(B10+B11)
    #   A: A01 coeff 1, A11 coeff -1
    #   B: B10 coeff 1, B11 coeff 1
    #   C: C00 coeff -1, C11 coeff 1
    
    # Let me define these as data structures.
    # A_blocks: for each j, list of (block_index, coeff) where block_index is (bi, bj)
    # B_blocks: for each j, list of (block_index, coeff) where block_index is (bj, bk)
    # C_blocks: for each j, list of (block_index, coeff) where block_index is (bi, bk)
    
    # Block index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Let's use flat index: bi*2 + bj for A, bj*2 + bk for B, bi*2 + bk for C
    
    A_block_coeffs = [
        [(0,1), (3,1)],  # j=0: A00, A11
        [(2,1), (3,1)],  # j=1: A10, A11
        [(0,1)],         # j=2: A00
        [(3,1)],         # j=3: A11
        [(0,1), (1,1)],  # j=4: A00, A01
        [(2,1), (0,-1)], # j=5: A10, A00
        [(1,1), (3,-1)]  # j=6: A01, A11
    ]
    
    B_block_coeffs = [
        [(0,1), (3,1)],  # j=0: B00, B11
        [(0,1)],         # j=1: B00
        [(1,1), (3,-1)], # j=2: B01, B11
        [(2,1), (0,-1)], # j=3: B10, B00
        [(3,1)],         # j=4: B11
        [(0,1), (1,1)],  # j=5: B00, B01
        [(2,1), (3,1)]   # j=6: B10, B11
    ]
    
    C_block_coeffs = [
        [(0,1), (3,1)],  # j=0: C00, C11
        [(2,1)],         # j=1: C10
        [(1,1)],         # j=2: C01
        [(0,1), (2,1)],  # j=3: C00, C10
        [(0,-1), (1,1)], # j=4: C00, C01
        [(3,1)],         # j=5: C11
        [(0,-1), (3,1)]  # j=6: C00, C11
    ]
    
    # Now, for each column r = j*7 + s:
    # U[row, r] = sum over (ab, coeff_ab) in A_block_coeffs[j] of coeff_ab * P_coeff[s][wi]
    #   where ab = bi*2 + bj, and row = 2*bi + wi, 2*bj + wj
    #   Wait, P_coeff[s] gives coefficients for (p1, p2, p3, p4) in the 2x2 matrix.
    #   p1 = element (0,0), p2 = element (0,1), p3 = element (1,0), p4 = element (1,1)
    #   So for block at position (bi, bj), element (wi, wj) has coefficient P_coeff[s][wi*2 + wj]
    
    # Let me map:
    # For a 2x2 matrix [[p1, p2], [p3, p4]]:
    # (0,0) -> p1, index 0
    # (0,1) -> p2, index 1
    # (1,0) -> p3, index 2
    # (1,1) -> p4, index 3
    
    # So for A block at (bi, bj), element (wi, wj) in the 4x4 matrix is at row 2*bi+wi, col 2*bj+wj.
    # The coefficient from P_coeff[s] is P_coeff[s][wi*2 + wj].
    
    # Similarly for B and C.
    
    # Let's build U, V, W.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(7):
        for s in range(7):
            r = j * 7 + s
            
            # Process A contributions
            for ab, coeff_ab in A_block_coeffs[j]:
                bi, bj = ab[0], ab[1]
                # ab is (block_index, coeff) where block_index is flat
                # Wait, I defined A_block_coeffs with tuples (block_index, coeff)
                # Let me re-examine.
            
            # Let me redefine more clearly.
            # A_block_coeffs[j] is a list of (block_flat_index, coeff)
            # where block_flat_index = bi*2 + bj for A, etc.
    
    # Let me redefine the data structures more clearly.
    
    # For A: blocks are indexed by (bi, bj), flat index = bi*2 + bj
    # For B: blocks are indexed by (bj, bk), flat index = bj*2 + bk
    # For C: blocks are indexed by (bi, bk), flat index = bi*2 + bk
    
    A_block_data = [
        [(0, 1), (3, 1)],   # j=0: A00 (0,0), A11 (1,1)
        [(2, 1), (3, 1)],   # j=1: A10 (1,0), A11 (1,1)
        [(0, 1)],           # j=2: A00 (0,0)
        [(3, 1)],           # j=3: A11 (1,1)
        [(0, 1), (1, 1)],   # j=4: A00 (0,0), A01 (0,1)
        [(2, 1), (0, -1)],  # j=5: A10 (1,0), A00 (0,0)
        [(1, 1), (3, -1)]   # j=6: A01 (0,1), A11 (1,1)
    ]
    
    B_block_data = [
        [(0, 1), (3, 1)],   # j=0: B00 (0,0), B11 (1,1)
        [(0, 1)],           # j=1: B00 (0,0)
        [(1, 1), (3, -1)],  # j=2: B01 (0,1), B11 (1,1)
        [(2, 1), (0, -1)],  # j=3: B10 (1,0), B00 (0,0)
        [(3, 1)],           # j=4: B11 (1,1)
        [(0, 1), (1, 1)],   # j=5: B00 (0,0), B01 (0,1)
        [(2, 1), (3, 1)]    # j=6: B10 (1,0), B11 (1,1)
    ]
    
    C_block_data = [
        [(0, 1), (3, 1)],   # j=0: C00 (0,0), C11 (1,1)
        [(2, 1)],           # j=1: C10 (1,0)
        [(1, 1)],           # j=2: C01 (0,1)
        [(0, 1), (2, 1)],   # j=3: C00 (0,0), C10 (1,0)
        [(0, -1), (1, 1)],  # j=4: C00 (0,0), C01 (0,1)
        [(3, 1)],           # j=5: C11 (1,1)
        [(0, -1), (3, 1)]   # j=6: C00 (0,0), C11 (1,1)
    ]
    
    # Now build U, V, W.
    for j in range(7):
        for s in range(7):
            r = j * 7 + s
            
            # U: A contributions
            for flat_ab, coeff_ab in A_block_data[j]:
                bi = flat_ab // 2
                bj = flat_ab % 2
                for wi in range(2):
                    for wj in range(2):
                        p_idx = wi * 2 + wj
                        row = 2 * bi + wi
                        col_in_block = wj
                        # The column in A is 2*bj + wj, but U is indexed by row only (flattened A)
                        # U has shape (n*m, rank) = (16, 49), indexed by (i, j) flattened
                        # i = 2*bi + wi, j = 2*bj + wj
                        flat_row = (2 * bi + wi) * 4 + (2 * bj + wj)
                        U[flat_row][r] += coeff_ab * P_coeff[s][p_idx]
            
            # V: B contributions
            for flat_bk, coeff_bk in B_block_data[j]:
                bj = flat_bk // 2
                bk = flat_bk % 2
                for wi in range(2):
                    for wj in range(2):
                        q_idx = wi * 2 + wj
                        row = 2 * bj + wi
                        col_in_block = wj
                        # V has shape (m*p, rank) = (16, 49), indexed by (j, k) flattened
                        # j = 2*bj + wi, k = 2*bk + wj
                        flat_row = (2 * bj + wi) * 4 + (2 * bk + wj)
                        V[flat_row][r] += coeff_bk * Q_coeff[s][q_idx]
            
            # W: C contributions
            # W[k*n + i, r] per the task convention
            # C is indexed by (bi, bk) blocks, within block (wi, wj)
            # C element at block (bi, bk), within (wi, wj) corresponds to row i = 2*bi+wi, col k = 2*bk+wj
            # In W, the index is k*n + i = (2*bk+wj)*4 + (2*bi+wi)
            for flat_c, coeff_c in C_block_data[j]:
                bi = flat_c // 2
                bk = flat_c % 2
                for wi in range(2):
                    for wj in range(2):
                        c_idx = wi * 2 + wj
                        # C_coeff[s][c_idx] gives coefficient for R at position (wi, wj)
                        # But wait, C_coeff gives coefficients for (R11, R12, R21, R22)
                        # R11 -> (0,0), R12 -> (0,1), R21 -> (1,0), R22 -> (1,1)
                        # So c_idx = wi*2 + wj is correct.
                        flat_row = (2 * bk + wj) * 4 + (2 * bi + wi)
                        W[flat_row][r] += coeff_c * C_coeff[s][c_idx]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
