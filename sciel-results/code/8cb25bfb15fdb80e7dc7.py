"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose 4x4x8 into two 4x4x4 multiplications.
    # Each 4x4x4 is decomposed using Strassen (rank 49).
    # Total rank = 98.
    
    # Strassen decomposition for 2x2x2 (rank 7)
    # Indices 0,1 for rows/cols of 2x2 block.
    # A is 2x2, B is 2x2, C is 2x2.
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to map these to the global 4x4x4 indices.
    # 4x4 A is split into 2x2 blocks:
    # A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # Each block is 2x2.
    # Let r,c be block indices (0,1) and i,j be local indices (0,1).
    # Global index for A: (r*2+i)*4 + (c*2+j)
    # Similarly for B and C.
    
    # Strassen for 4x4x4:
    # Let A_blocks be A00, A01, A10, A11.
    # Let B_blocks be B00, B01, B10, B11.
    # Products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # Each Pk is a 2x2 matrix product.
    # We decompose each Pk using the 2x2x2 Strassen terms.
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val

    # Define the 7 Strassen coefficients for 2x2x2
    # Each entry is (u_coeffs, v_coeffs, w_coeffs) for the 4 entries of the 2x2 matrices
    # u_coeffs: dict mapping (i,j) to coeff
    # We need to generate the 7 terms for a given 2x2x2 multiplication.
    
    def get_2x2_strassen_terms():
        # Returns list of 7 tuples (u, v, w) where u,v,w are dicts {(i,j): coeff}
        terms = []
        # M1
        u1 = {(0,0): 1, (1,1): 1}
        v1 = {(0,0): 1, (1,1): 1}
        w1 = {(0,0): 1, (0,1): 0, (1,0): 0, (1,1): 0} # Contribution to C00? No, w is the output coeff.
        # Actually, the term is u^T A * v^T B -> w^T C
        # So w is the vector of coeffs for C.
        # For M1, it contributes to C00, C01, C10, C11?
        # No, M1 is a scalar product of two linear forms?
        # In 2x2x2, A and B are 2x2.
        # M1 = (A00+A11)(B00+B11) is a scalar.
        # This scalar is added to C00, C11?
        # C00 = M1 + ...
        # C11 = M1 - ...
        # So M1 contributes 1 to C00 and 1 to C11?
        # Wait, the formula is:
        # C00 = M1 + M4 - M5 + M7
        # C11 = M1 - M2 + M3 + M6
        # So M1 contributes to C00 with coeff 1 and C11 with coeff 1.
        # What about C01 and C10? They don't involve M1.
        # So w1 = {(0,0): 1, (1,1): 1}
        
        # Let's list the w coeffs for each Mk
        # M1: C00(1), C11(1)
        # M2: C10(1), C11(-1)
        # M3: C01(1), C11(1)
        # M4: C00(1), C10(1)
        # M5: C00(-1), C01(1)
        # M6: C11(1)
        # M7: C00(1)
        
        w_coeffs = [
            {(0,0): 1, (1,1): 1}, # M1
            {(1,0): 1, (1,1): -1}, # M2
            {(0,1): 1, (1,1): 1}, # M3
            {(0,0): 1, (1,0): 1}, # M4
            {(0,0): -1, (0,1): 1}, # M5
            {(1,1): 1}, # M6
            {(0,0): 1} # M7
        ]
        
        u_coeffs = [
            {(0,0): 1, (1,1): 1}, # M1
            {(1,0): 1, (1,1): 1}, # M2
            {(0,0): 1}, # M3
            {(1,1): 1}, # M4
            {(0,0): 1, (0,1): 1}, # M5
            {(1,0): 1, (0,0): -1}, # M6
            {(0,1): 1, (1,1): -1} # M7
        ]
        
        v_coeffs = [
            {(0,0): 1, (1,1): 1}, # M1
            {(0,0): 1}, # M2
            {(0,1): 1, (1,1): -1}, # M3
            {(1,0): 1, (0,0): -1}, # M4
            {(1,1): 1}, # M5
            {(0,0): 1, (0,1): 1}, # M6
            {(1,0): 1, (1,1): 1} # M7
        ]
        
        return list(zip(u_coeffs, v_coeffs, w_coeffs))

    base_terms = get_2x2_strassen_terms()
    
    # Now apply to 4x4x4
    # 7 Strassen products P1..P7
    # Each Pk is a product of two 2x2 matrices.
    # We need to expand each Pk into 7 terms using base_terms.
    
    # P1 = (A00+A11)(B00+B11)
    # The "A" part is A00+A11. The "B" part is B00+B11.
    # We treat A00+A11 as a 2x2 matrix where entry (i,j) is A00[i,j] + A11[i,j].
    # Similarly for B.
    
    # We will iterate over the 7 Strassen products for the 4x4 level.
    # For each product, we iterate over the 7 Strassen terms for the 2x2 level.
    
    col = 0
    # Strassen 4x4 products definitions
    # Each product is defined by u_block_coeffs and v_block_coeffs
    # u_block_coeffs: dict mapping block (r,c) to coeff
    # v_block_coeffs: dict mapping block (r,c) to coeff
    
    p1_u = {(0,0): 1, (1,1): 1}
    p1_v = {(0,0): 1, (1,1): 1}
    
    p2_u = {(1,0): 1, (1,1): 1}
    p2_v = {(0,0): 1}
    
    p3_u = {(0,0): 1}
    p3_v = {(0,1): 1, (1,1): -1}
    
    p4_u = {(1,1): 1}
    p4_v = {(1,0): 1, (0,0): -1}
    
    p5_u = {(0,0): 1, (0,1): 1}
    p5_v = {(1,1): 1}
    
    p6_u = {(1,0): 1, (0,0): -1}
    p6_v = {(0,0): 1, (0,1): 1}
    
    p7_u = {(0,1): 1, (1,1): -1}
    p7_v = {(1,0): 1, (1,1): 1}
    
    products = [
        (p1_u, p1_v),
        (p2_u, p2_v),
        (p3_u, p3_v),
        (p4_u, p4_v),
        (p5_u, p5_v),
        (p6_u, p6_v),
        (p7_u, p7_v)
    ]
    
    # C reconstruction for 4x4
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We need to add the contributions of each Pk to the global W matrix.
    # For a given Pk, we have 7 terms from base_terms.
    # Each term has u_local, v_local, w_local (for 2x2 blocks).
    # The global U index for A entry in block (br, bc) at local (i,j) is (br*2+i)*4 + (bc*2+j).
    # The global V index for B entry in block (br, bc) at local (i,j) is (br*2+i)*4 + (bc*2+j).
    # The global W index for C entry in block (br, bc) at local (i,j) is (br*2+i)*4 + (bc*2+j).
    
    # We will process each of the 7 products Pk.
    for pk_u, pk_v in products:
        # For each of the 7 base terms
        for u_local, v_local, w_local in base_terms:
            # u_local is a dict of local (i,j) -> coeff for the "A" part of this term
            # But the "A" part of this term is a linear combination of blocks defined by pk_u.
            # So the global U coeff for A[global_i, global_j] is:
            # Sum over blocks (br,bc) in pk_u: pk_u[(br,bc)] * u_local.get((local_i, local_j), 0)
            # where global_i = br*2+local_i, global_j = bc*2+local_j
            
            # Similarly for V and W.
            
            # We add this to the current column col.
            
            # Initialize local accumulators for this rank-1 term
            u_global = {}
            v_global = {}
            w_global = {}
            
            # Fill U
            for (br, bc), coeff_u_block in pk_u.items():
                for (li, lj), coeff_u_local in u_local.items():
                    gi = br * 2 + li
                    gj = bc * 2 + lj
                    idx = gi * 4 + gj
                    u_global[idx] = u_global.get(idx, 0) + coeff_u_block * coeff_u_local
            
            # Fill V
            for (br, bc), coeff_v_block in pk_v.items():
                for (li, lj), coeff_v_local in v_local.items():
                    gi = br * 2 + li
                    gj = bc * 2 + lj
                    idx = gi * 4 + gj
                    v_global[idx] = v_global.get(idx, 0) + coeff_v_block * coeff_v_local
            
            # Fill W
            # The W part depends on which C block this Pk contributes to.
            # We need to know the coefficient of Pk in each C block.
            # This is determined by the Strassen formula for 4x4.
            # We are iterating over products P1..P7.
            # We need to pass the w_block_coeffs for this Pk.
            
            # Let's restructure: iterate over Pk, get w_block_coeffs for Pk.
            pass

    # Restructure loop to include W block coeffs
    col = 0
    # Define W block coeffs for each Pk
    w_block_coeffs_list = [
        {(0,0): 1, (1,1): 1}, # P1
        {(1,0): 1, (1,1): -1}, # P2
        {(0,1): 1, (1,1): 1}, # P3
        {(0,0): 1, (1,0): 1}, # P4
        {(0,0): -1, (0,1): 1}, # P5
        {(1,1): 1}, # P6
        {(0,0): 1} # P7
    ]
    
    for idx_p, (pk_u, pk_v) in enumerate(products):
        w_block_coeffs = w_block_coeffs_list[idx_p]
        
        for u_local, v_local, w_local in base_terms:
            u_global = {}
            v_global = {}
            w_global = {}
            
            for (br, bc), coeff_u_block in pk_u.items():
                for (li, lj), coeff_u_local in u_local.items():
                    gi = br * 2 + li
                    gj = bc * 2 + lj
                    idx = gi * 4 + gj
                    u_global[idx] = u_global.get(idx, 0) + coeff_u_block * coeff_u_local
            
            for (br, bc), coeff_v_block in pk_v.items():
                for (li, lj), coeff_v_local in v_local.items():
                    gi = br * 2 + li
                    gj = bc * 2 + lj
                    idx = gi * 4 + gj
                    v_global[idx] = v_global.get(idx, 0) + coeff_v_block * coeff_v_local
            
            # W is for the 4x4x4 left half (k=0..3)
            # Global W index: k*n + i = k*4 + i
            # But W is (p*n, rank) = (32, rank).
            # k is 0..3 for left half.
            # C block (br, bc) has local (li, lj).
            # Global i = br*2+li, Global k = bc*2+lj.
            # W index = k*4 + i.
            
            for (br, bc), coeff_w_block in w_block_coeffs.items():
                for (li, lj), coeff_w_local in w_local.items():
                    gi = br * 2 + li
                    gk = bc * 2 + lj
                    idx = gk * 4 + gi
                    w_global[idx] = w_global.get(idx, 0) + coeff_w_block * coeff_w_local
            
            # Add to U, V, W
            for idx, val in u_global.items():
                U[idx][col] += val
            for idx, val in v_global.items():
                V[idx][col] += val
            for idx, val in w_global.items():
                W[idx][col] += val
            
            col += 1
            
    # Now do the same for the right half (k=4..7)
    # The A and B parts are the same?
    # No, B is different.
    # B is 4x8. Left half is B[0..3], Right half is B[4..7].
    # So for the right half, we use the same U (A) coefficients.
    # But V coefficients are for the right half of B.
    # V index: j*p + k = j*8 + k.
    # For left half, k=0..3. For right half, k=4..7.
    # The structure of B_right is the same as B_left.
    # So we can reuse the same V local/block coeffs, but map to different global V indices.
    # And W maps to k=4..7.
    
    for idx_p, (pk_u, pk_v) in enumerate(products):
        w_block_coeffs = w_block_coeffs_list[idx_p]
        
        for u_local, v_local, w_local in base_terms:
            u_global = {}
            v_global = {}
            w_global = {}
            
            for (br, bc), coeff_u_block in pk_u.items():
                for (li, lj), coeff_u_local in u_local.items():
                    gi = br * 2 + li
                    gj = bc * 2 + lj
                    idx = gi * 4 + gj
                    u_global[idx] = u_global.get(idx, 0) + coeff_u_block * coeff_u_local
            
            # V global index: j*8 + k
            # B block (br, bc) -> j = br*2+li, k = bc*2+lj + 4 (offset for right half)
            for (br, bc), coeff_v_block in pk_v.items():
                for (li, lj), coeff_v_local in v_local.items():
                    gj = br * 2 + li
                    gk = bc * 2 + lj + 4
                    idx = gj * 8 + gk
                    v_global[idx] = v_global.get(idx, 0) + coeff_v_block * coeff_v_local
            
            # W global index: k*4 + i
            # C block (br, bc) -> i = br*2+li, k = bc*2+lj + 4
            for (br, bc), coeff_w_block in w_block_coeffs.items():
                for (li, lj), coeff_w_local in w_local.items():
                    gi = br * 2 + li
                    gk = bc * 2 + lj + 4
                    idx = gk * 4 + gi
                    w_global[idx] = w_global.get(idx, 0) + coeff_w_block * coeff_w_local
            
            for idx, val in u_global.items():
                U[idx][col] += val
            for idx, val in v_global.items():
                V[idx][col] += val
            for idx, val in w_global.items():
                W[idx][col] += val
            
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
