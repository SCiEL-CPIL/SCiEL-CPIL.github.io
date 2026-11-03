"""Matrix multiplication 6x6x6 using 2x2 block decomposition with rank-21 3x3 decomposition."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 6, 6, 6
    
    # We use a 2x2 block decomposition where each block is 3x3.
    # The 2x2 block multiplication requires 8 distinct 3x3 matrix multiplications:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # We need a rank-21 decomposition for 3x3 matrix multiplication.
    # The tensor rank of 3x3 matrix multiplication is known to be 19 (Laderman),
    # but let's use a valid decomposition.
    
    # For 3x3, define the decomposition with rank r3.
    # U33[i*3+j, c], V33[j*3+k, c], W33[k*3+i, c]
    # C[i,k] = sum_c U33[i*3+j, c] * V33[j*3+k, c] * W33[k*3+i, c]
    #         = sum_j A[i,j] * B[j,k]
    
    # Let's use the schoolbook for 3x3 as a fallback (rank 27), but that gives 8*27=216.
    # Instead, let's try to use a rank-19 Laderman decomposition.
    
    # Laderman's rank-19 algorithm for 3x3:
    # The 19 bilinear forms are:
    # Let's index them properly.
    
    # Actually, the known minimal rank for 3x3 is 19. Let me implement it carefully.
    # Reference: Laderman (1976) "A non-recursive matrix multiplication algorithm"
    
    # For 3x3 matrices A = (a_ij) and B = (b_jk), C = A*B:
    # The 19 products are:
    # m0  = a00*b00
    # m1  = a11*b11
    # m2  = a22*b22
    # m3  = a01*b10
    # m4  = a12*b20
    # m5  = a20*b01
    # m6  = a02*b21
    # m7  = a10*b02
    # m8  = a21*b12
    # m9  = (a00+a10)*(b00+b10)
    # m10 = (a01+a11)*(b10+b11)
    # m11 = (a02+a12)*(b20+b21)
    # m12 = (a00+a20)*(b00+b20)
    # m13 = (a01+a21)*(b10+b20)
    # m14 = (a02+a22)*(b20+b21)  -- wait, this seems off
    # m15 = (a10+a20)*(b01+b21)
    # m16 = (a11+a21)*(b11+b21)
    # m17 = (a12+a22)*(b21+b22)
    # m18 = (a00+a01+a02)*(b00+b10+b20)
    
    # This is getting error-prone from memory. Let me use a different approach.
    # I'll use the fact that we can verify correctness by checking the tensor equation.
    
    # Let me use a simpler approach: use the rank-27 schoolbook for 3x3,
    # which gives total rank 8*27 = 216 (same as current). That's not an improvement.
    
    # Let me try a different strategy: use the 2x2 Strassen decomposition at the block level.
    # Strassen for 2x2 block multiplication uses 7 multiplications of 3x3 blocks.
    # If each 3x3 multiplication has rank r, total rank = 7*r.
    # With schoolbook 3x3 (r=27): 7*27 = 189 < 216. That's an improvement!
    
    # Strassen's algorithm for 2x2 block matrices:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So we need 7 rank-r decompositions for 3x3.
    # Using schoolbook (r=27): total rank = 7*27 = 189.
    
    # Let's implement this.
    
    # First, build the rank-27 schoolbook decomposition for 3x3.
    # U33[i*3+j, c], V33[j*3+k, c], W33[k*3+i, c]
    # For schoolbook: each (i,j,k) triple is one rank-1 term.
    # U33[i*3+j, i*9+j*3+k] = 1
    # V33[j*3+k, i*9+j*3+k] = 1
    # W33[k*3+i, i*9+j*3+k] = 1
    
    r3 = 27  # rank for 3x3 schoolbook
    
    # Build U33, V33, W33 as 9 x r3 matrices
    U33 = [[0] * r3 for _ in range(9)]
    V33 = [[0] * r3 for _ in range(9)]
    W33 = [[0] * r3 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U33[i*3+j][col] = 1
                V33[j*3+k][col] = 1
                W33[k*3+i][col] = 1
                col += 1
    
    # Now build the full 6x6 decomposition using Strassen at the block level.
    # Total rank = 7 * r3 = 7 * 27 = 189
    
    rank = 7 * r3
    U = [[0] * rank for _ in range(36)]  # 6*6
    V = [[0] * rank for _ in range(36)]  # 6*6
    W = [[0] * rank for _ in range(36)]  # 6*6
    
    # Block indices:
    # A = [[A00, A01], [A10, A11]]
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    #
    # A00 covers rows 0-2, cols 0-2 of A
    # A01 covers rows 0-2, cols 3-5 of A
    # A10 covers rows 3-5, cols 0-2 of A
    # A11 covers rows 3-5, cols 3-5 of A
    # Similarly for B and C.
    
    # For each Strassen multiplication M_m (m=0..6), we need to:
    # 1. Form the linear combinations of A-blocks and B-blocks
    # 2. Apply the 3x3 schoolbook decomposition
    # 3. Add contributions to C-blocks
    
    # Strassen multiplications:
    # M0 = (A00 + A11) * (B00 + B11)
    # M1 = (A10 + A11) * B00
    # M2 = A00 * (B01 - B11)
    # M3 = A11 * (B10 - B00)
    # M4 = (A00 + A01) * B11
    # M5 = (A10 - A00) * (B00 + B01)
    # M6 = (A01 - A11) * (B10 + B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # For each Strassen term, define:
    # - Which A-blocks contribute with what coefficients
    # - Which B-blocks contribute with what coefficients
    # - Which C-blocks receive contributions with what coefficients
    
    # A-block to row range mapping:
    # A00: rows 0-2, cols 0-2 -> U rows 0-8 (i*6+j for i in 0-2, j in 0-2)
    # A01: rows 0-2, cols 3-5 -> U rows 3-5, 6-8, 9-11? No...
    # A is 6x6, so A[i,j] is at U[i*6+j]
    # A00: i in 0-2, j in 0-2 -> U[i*6+j] for i in 0-2, j in 0-2
    # A01: i in 0-2, j in 3-5 -> U[i*6+j] for i in 0-2, j in 3-5
    # A10: i in 3-5, j in 0-2 -> U[i*6+j] for i in 3-5, j in 0-2
    # A11: i in 3-5, j in 3-5 -> U[i*6+j] for i in 3-5, j in 3-5
    
    # B is 6x6, B[j,k] is at V[j*6+k]
    # B00: j in 0-2, k in 0-2 -> V[j*6+k] for j in 0-2, k in 0-2
    # B01: j in 0-2, k in 3-5 -> V[j*6+k] for j in 0-2, k in 3-5
    # B10: j in 3-5, k in 0-2 -> V[j*6+k] for j in 3-5, k in 0-2
    # B11: j in 3-5, k in 3-5 -> V[j*6+k] for j in 3-5, k in 3-5
    
    # C is 6x6, C[i,k] is at W[k*6+i]
    # C00: i in 0-2, k in 0-2 -> W[k*6+i] for i in 0-2, k in 0-2
    # C01: i in 0-2, k in 3-5 -> W[k*6+i] for i in 0-2, k in 3-5
    # C10: i in 3-5, k in 0-2 -> W[k*6+i] for i in 3-5, k in 0-2
    # C11: i in 3-5, k in 3-5 -> W[k*6+i] for i in 3-5, k in 3-5
    
    # Define the Strassen terms:
    # Each term m has:
    #   a_coeffs: dict mapping block name to coefficient for A
    #   b_coeffs: dict mapping block name to coefficient for B
    #   c_coeffs: dict mapping block name to coefficient for C
    
    strassen_terms = [
        # M0 = (A00 + A11) * (B00 + B11)
        {
            'a': {'A00': 1, 'A11': 1},
            'b': {'B00': 1, 'B11': 1},
            'c': {'C00': 1, 'C11': 1}
        },
        # M1 = (A10 + A11) * B00
        {
            'a': {'A10': 1, 'A11': 1},
            'b': {'B00': 1},
            'c': {'C10': 1, 'C11': -1}
        },
        # M2 = A00 * (B01 - B11)
        {
            'a': {'A00': 1},
            'b': {'B01': 1, 'B11': -1},
            'c': {'C01': 1, 'C11': 1}
        },
        # M3 = A11 * (B10 - B00)
        {
            'a': {'A11': 1},
            'b': {'B10': 1, 'B00': -1},
            'c': {'C00': 1, 'C10': 1}
        },
        # M4 = (A00 + A01) * B11
        {
            'a': {'A00': 1, 'A01': 1},
            'b': {'B11': 1},
            'c': {'C00': -1, 'C01': 1}
        },
        # M5 = (A10 - A00) * (B00 + B01)
        {
            'a': {'A10': 1, 'A00': -1},
            'b': {'B00': 1, 'B01': 1},
            'c': {'C11': 1}
        },
        # M6 = (A01 - A11) * (B10 + B11)
        {
            'a': {'A01': 1, 'A11': -1},
            'b': {'B10': 1, 'B11': 1},
            'c': {'C00': 1}
        },
    ]
    
    # Block to index mapping:
    # For A: block (bi, bj) where bi, bj in {0,1}
    #   bi=0: rows 0-2, bi=1: rows 3-5
    #   bj=0: cols 0-2, bj=1: cols 3-5
    #   U index for A[bi*3+i, bj*3+j] = (bi*3+i)*6 + (bj*3+j)
    
    # For B: block (bk, bl) where bk, bl in {0,1}
    #   bk=0: rows 0-2, bk=1: rows 3-5
    #   bl=0: cols 0-2, bl=1: cols 3-5
    #   V index for B[bk*3+j, bl*3+k] = (bk*3+j)*6 + (bl*3+k)
    
    # For C: block (ci, ck) where ci, ck in {0,1}
    #   ci=0: rows 0-2, ci=1: rows 3-5
    #   ck=0: cols 0-2, ck=1: cols 3-5
    #   W index for C[ci*3+i, ck*3+k] = (ck*3+k)*6 + (ci*3+i)
    
    # Block name to (bi, bj) mapping:
    block_map = {
        'A00': (0, 0), 'A01': (0, 1), 'A10': (1, 0), 'A11': (1, 1),
        'B00': (0, 0), 'B01': (0, 1), 'B10': (1, 0), 'B11': (1, 1),
        'C00': (0, 0), 'C01': (0, 1), 'C10': (1, 0), 'C11': (1, 1),
    }
    
    # For each Strassen term, for each rank-1 component of the 3x3 schoolbook,
    # add contributions to U, V, W.
    
    for m_idx, term in enumerate(strassen_terms):
        base_col = m_idx * r3
        
        for local_col in range(r3):
            col = base_col + local_col
            
            # Get the 3x3 schoolbook indices for this local column
            # local_col = i*9 + j*3 + k for some i,j,k in 0-2
            i3 = local_col // 9
            rem = local_col % 9
            j3 = rem // 3
            k3 = rem % 3
            
            # For A: for each block with coefficient ca
            for a_block, ca in term['a'].items():
                bi, bj = block_map[a_block]
                # A[bi*3+i3, bj*3+j3] is at U[(bi*3+i3)*6 + (bj*3+j3)]
                u_row = (bi*3+i3)*6 + (bj*3+j3)
                U[u_row][col] += ca * U33[i3*3+j3][local_col]
            
            # For B: for each block with coefficient cb
            for b_block, cb in term['b'].items():
                bk, bl = block_map[b_block]
                # B[bk*3+j3, bl*3+k3] is at V[(bk*3+j3)*6 + (bl*3+k3)]
                v_row = (bk*3+j3)*6 + (bl*3+k3)
                V[v_row][col] += cb * V33[j3*3+k3][local_col]
            
            # For C: for each block with coefficient cc
            for c_block, cc in term['c'].items():
                ci, ck = block_map[c_block]
                # C[ci*3+i3, ck*3+k3] is at W[(ck*3+k3)*6 + (ci*3+i3)]
                w_row = (ck*3+k3)*6 + (ci*3+i3)
                W[w_row][col] += cc * W33[k3*3+i3][local_col]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
