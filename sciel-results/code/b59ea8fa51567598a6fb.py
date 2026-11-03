def construct():
    n, m, p = 4, 4, 6
    
    # Schoolbook fallback
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1

    # Strassen-like for 4x4 * 4x6
    # Blocks: A is 2x2 blocks of 2x2. B is 2x2 blocks of 2x3. C is 2x2 blocks of 2x3.
    # Inner multiplication: 2x2 * 2x3. Rank 12 (schoolbook).
    # Total rank: 7 * 12 = 84.
    
    rank = 84
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    current_rank = 0
    
    # Helper to get global indices for A block (br, bc) and local (u, v)
    # A is 4x4. Block size 2x2.
    # br in {0,1}, bc in {0,1}
    def a_idx(br, bc, u, v):
        i = 2 * br + u
        j = 2 * bc + v
        return i * m + j

    # Helper to get global indices for B block (br, bc) and local (u, v)
    # B is 4x6. Block size 2x3.
    # br in {0,1}, bc in {0,1}
    # u in {0,1}, v in {0,1,2}
    def b_idx(br, bc, u, v):
        j = 2 * br + u  # Row in B corresponds to Col in A
        k = 3 * bc + v  # Col in B
        return j * p + k

    # Helper to get global indices for C block (br, bc) and local (u, v)
    # C is 4x6. Block size 2x3.
    # br in {0,1}, bc in {0,1}
    # u in {0,1}, v in {0,1,2}
    def c_idx(br, bc, u, v):
        i = 2 * br + u  # Row in C corresponds to Row in A
        k = 3 * bc + v  # Col in C corresponds to Col in B
        return k * n + i

    def add_inner_mult(a_br, a_bc, a_coeff, b_br, b_bc, b_coeff):
        nonlocal current_rank
        # a_br, a_bc: block coords in A
        # b_br, b_bc: block coords in B
        # Note: For the product to be non-zero, a_bc must match b_br (inner dimension)
        if a_bc != b_br:
            return
            
        # c_br must match a_br, c_bc must match b_bc
        c_br = a_br
        c_bc = b_bc
        
        # Iterate over local indices
        for u in range(2): # row of A block / row of C block
            for v in range(2): # col of A block / row of B block
                for w in range(3): # col of B block / col of C block
                    u_idx = a_idx(a_br, a_bc, u, v)
                    v_idx = b_idx(b_br, b_bc, v, w)
                    w_idx = c_idx(c_br, c_bc, u, w)
                    
                    U[u_idx][current_rank] = a_coeff
                    V[v_idx][current_rank] = b_coeff
                    W[w_idx][current_rank] = 1
                    
                    current_rank += 1

    # Strassen Multiplications
    # M1 = (A11 + A22)(B11 + B22)
    add_inner_mult(0, 0, 1, 0, 0, 1) # A11 * B11
    add_inner_mult(0, 0, 1, 0, 1, 1) # A11 * B22 (Wait, B22 is block (1,1)? No.
    # Let's map block IDs:
    # A00 (br=0, bc=0), A01 (br=0, bc=1), A10 (br=1, bc=0), A11 (br=1, bc=1)
    # B00 (br=0, bc=0), B01 (br=0, bc=1), B10 (br=1, bc=0), B11 (br=1, bc=1)
    #
    # M1 = (A00 + A11)(B00 + B11)
    # Terms: A00*B00, A00*B11, A11*B00, A11*B11
    # Note: A00 (bc=0) * B11 (br=1) -> 0 != 1, so zero.
    # A11 (bc=1) * B00 (br=0) -> 1 != 0, so zero.
    # So only A00*B00 and A11*B11 are non-zero?
    # Wait. Strassen formula:
    # M1 = (A11 + A22)(B11 + B22) in 1-based indexing for 2x2 blocks.
    # Let's stick to 0-based:
    # M1 = (A00 + A11)(B00 + B11)
    # Expansion: A00*B00 + A00*B11 + A11*B00 + A11*B11.
    # A00 is top-left (rows 0-1, cols 0-1). B00 is top-left (rows 0-1, cols 0-2).
    # A00 * B00: A cols 0-1, B rows 0-1. Match.
    # A00 * B11: B11 is bottom-right (rows 2-3, cols 3-5). A cols 0-1, B rows 2-3. Mismatch. Zero.
    # A11 * B00: A11 is bottom-right (rows 2-3, cols 2-3). B00 is top-left (rows 0-1). Mismatch. Zero.
    # A11 * B11: A cols 2-3, B rows 2-3. Match.
    #
    # So M1 effectively only computes A00*B00 and A11*B11?
    # That's not Strassen. Strassen relies on the sums to create cross terms that are NOT zero in the block multiplication sense?
    # No, Strassen works on the *scalar* level of the blocks.
    # The block multiplication (A00+A11)(B00+B11) is only valid if the inner dimensions match.
    # Here, A00 is 2x2, A11 is 2x2. Sum is 2x2.
    # B00 is 2x3, B11 is 2x3. Sum is 2x3.
    # Product is 2x3.
    # But does A00*B11 contribute?
    # In the full matrix product, C = AB.
    # C00 = A00*B00 + A01*B10.
    # C01 = A00*B01 + A01*B11.
    # C10 = A10*B00 + A11*B10.
    # C11 = A10*B01 + A11*B11.
    #
    # Strassen M1 = (A00 + A11)(B00 + B11).
    # This sum is NOT a block of C. It's an auxiliary matrix.
    # The formula for C blocks uses M1, M2, etc.
    # C00 = M1 + M4 - M5 + M7.
    #
    # The "Zero" logic in my `add_inner_mult` checks if the specific block pair (a_bc, b_br) matches.
    # For M1, we are computing the tensor for the product of the sum (A00+A11) and (B00+B11).
    # This involves 4 terms: A00*B00, A00*B11, A11*B00, A11*B11.
    # A00*B11: A00 cols 0-1. B11 rows 2-3. In the full matrix, this product is zero.
    # So we only need to add the non-zero terms.
    #
    # So for M1:
    # add_inner_mult(0, 0, 1, 0, 0, 1) # A00 * B00
    # add_inner_mult(1, 1, 1, 1, 1, 1) # A11 * B11
    # The cross terms are zero.
    
    # Let's re-verify the Strassen formulas with this "zero cross term" property.
    # If cross terms are zero, Strassen degenerates?
    # No, Strassen is for square matrices where all block products are non-zero.
    # Here, the block structure is such that A00*B11 is zero.
    # This means the "Strassen" approach might not reduce the count as much, or the formulas need adjustment.
    #
    # Actually, if A00*B11 is zero, then M1 = A00*B00 + A11*B11.
    # M2 = (A10 + A11)B00 = A10*B00 + A11*B00 (A11*B00 is zero) = A10*B00.
    # M3 = A00(B01 - B11) = A00*B01 - A00*B11 (zero) = A00*B01.
    # M4 = A11(B10 - B00) = A11*B10 - A11*B00 (zero) = A11*B10.
    # M5 = (A00 + A01)B11 = A00*B11 (zero) + A01*B11 = A01*B11.
    # M6 = (A10 - A00)(B00 + B01) = (A10*B00 + A10*B01) - (A00*B00 + A00*B01).
    # M7 = (A01 - A11)(B10 + B11) = (A01*B10 + A01*B11) - (A11*B10 + A11*B11).
    
    # Let's check C00 = M1 + M4 - M5 + M7
    # = (A00*B00 + A11*B11) + (A11*B10) - (A01*B11) + (A01*B10 + A01*B11 - A11*B10 - A11*B11)
    # = A00*B00 + A11*B11 + A11*B10 - A01*B11 + A01*B10 + A01*B11 - A11*B10 - A11*B11
    # = A00*B00 + A01*B10.
    # Correct!
    
    # So the strategy works. We just skip zero terms.
    
    # M1
    add_inner_mult(0, 0, 1, 0, 0, 1)
    add_inner_mult(1, 1, 1, 1, 1, 1)
    
    # M2 = (A10 + A11)B00 -> A10*B00
    add_inner_mult(1, 0, 1, 0, 0, 1)
    
    # M3 = A00(B01 - B11) -> A00*B01
    add_inner_mult(0, 0, 1, 0, 1, 1)
    
    # M4 = A11(B10 - B00) -> A11*B10
    add_inner_mult(1, 1, 1, 1, 0, 1)
    
    # M5 = (A00 + A01)B11 -> A01*B11
    add_inner_mult(0, 1, 1, 1, 1, 1)
    
    # M6 = (A10 - A00)(B00 + B01)
    # = A10*B00 + A10*B01 - A00*B00 - A00*B01
    add_inner_mult(1, 0, 1, 0, 0, 1)
    add_inner_mult(1, 0, 1, 0, 1, 1)
    add_inner_mult(0, 0, -1, 0, 0, 1)
    add_inner_mult(0, 0, -1, 0, 1, 1)
    
    # M7 = (A01 - A11)(B10 + B11)
    # = A01*B10 + A01*B11 - A11*B10 - A11*B11
    add_inner_mult(0, 1, 1, 1, 0, 1)
    add_inner_mult(0, 1, 1, 1, 1, 1)
    add_inner_mult(1, 1, -1, 1, 0, 1)
    add_inner_mult(1, 1, -1, 1, 1, 1)
    
    if current_rank != rank:
        return {"U": U_school, "V": V_school, "W": W_school}
        
    return {"U": U, "V": V, "W": W}
