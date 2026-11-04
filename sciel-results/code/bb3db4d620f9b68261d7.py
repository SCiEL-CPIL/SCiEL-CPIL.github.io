"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    U = []
    V = []
    W = []
    
    # Helper to add a rank-1 term
    def add_term(u_vec, v_vec, w_vec):
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)

    # --- Part 1: 4x4x4 decomposition using Strassen (Rank 49) ---
    
    # Inner 2x2x2 Strassen decomposition (Rank 7)
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # 7 products:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # u, v, w for the 7 terms of 2x2x2
    # u, v are length 4 (a,b,c,d and e,f,g,h)
    # w is length 4 (C00, C01, C10, C11)
    
    inner_strassen = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]),
        # M3
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 1, 0, 0]),
        # M4
        ([0, 0, 0, 1], [0, 1, -1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 1])
    ]
    
    # Outer 4x4x4 Strassen decomposition (Rank 7 block multiplications)
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    # 7 products:
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B10-B11)
    # P4 = A11(B01-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We represent A blocks as indices 0-3 in the outer structure.
    # A_flat: A00 (0,1,4,5), A01 (2,3,6,7), A10 (8,9,12,13), A11 (10,11,14,15)
    # B_flat: B00 (0,1,4,5), B01 (2,3,6,7), B10 (8,9,12,13), B11 (10,11,14,15)
    # C_flat: C00 (0,1,4,5), C01 (2,3,6,7), C10 (8,9,12,13), C11 (10,11,14,15)
    
    # Outer combinations (coeffs for A blocks, coeffs for B blocks, coeffs for C blocks)
    # Each block is 2x2 = 4 elements.
    # u_outer is length 16, v_outer is length 16, w_outer is length 16.
    
    outer_strassen = [
        # P1
        ([1,1,0,0], [1,0,0,1], [1,0,0,0]), # (A00+A11) * (B00+B11) -> C00
        # P2
        ([0,0,1,1], [1,0,0,0], [0,0,1,0]), # (A10+A11) * B00 -> C10
        # P3
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]), # A00 * (B10-B11) -> C01
        # P4
        ([0,0,0,1], [1,-1,0,0], [1,0,1,0]), # A11 * (B01-B00) -> C00, C10
        # P5
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]), # (A00+A01) * B11 -> C00, C01
        # P6
        ([0,0,1,-1], [1,1,0,0], [0,0,0,1]), # (A10-A00) * (B00+B01) -> C11
        # P7
        ([0,1,0,-1], [0,0,1,1], [1,0,0,1])  # (A01-A11) * (B10+B11) -> C00, C11
    ]
    
    # Note: The w_outer in the list above is for the *block* C.
    # We need to expand these block coefficients into the full 16-dim vector.
    # And combine with the inner 2x2x2 decomposition.
    
    for u_out_blocks, v_out_blocks, w_out_blocks in outer_strassen:
        # Expand u_out_blocks (4 blocks) to u_out (16 elements)
        u_out = [0]*16
        for b_idx in range(4):
            coeff = u_out_blocks[b_idx]
            if coeff != 0:
                # Block b_idx covers indices:
                # 0: 0,1,4,5
                # 1: 2,3,6,7
                # 2: 8,9,12,13
                # 3: 10,11,14,15
                idxs = [b_idx*2, b_idx*2+1, b_idx*2+4, b_idx*2+5]
                for idx in idxs:
                    u_out[idx] = coeff
        
        # Expand v_out_blocks (4 blocks) to v_out (16 elements)
        v_out = [0]*16
        for b_idx in range(4):
            coeff = v_out_blocks[b_idx]
            if coeff != 0:
                idxs = [b_idx*2, b_idx*2+1, b_idx*2+4, b_idx*2+5]
                for idx in idxs:
                    v_out[idx] = coeff
        
        # Now combine with inner 2x2x2 decomposition
        # The outer product is a 2x2 matrix multiplication of the blocks.
        # We use the 7 inner rank-1 terms to decompose this 2x2 multiplication.
        for u_in, v_in, w_in in inner_strassen:
            # u_in is length 4 (local 2x2 A)
            # v_in is length 4 (local 2x2 B)
            # w_in is length 4 (local 2x2 C)
            
            # The full U vector is u_out * u_in?
            # No. The outer u_out is a linear combination of A blocks.
            # The inner u_in is a linear combination of the elements of that block.
            # So the full linear combination is the product of the coefficients.
            
            full_u = [0]*16
            for b_idx in range(4):
                coeff_out = u_out_blocks[b_idx]
                if coeff_out == 0: continue
                # u_in corresponds to the 4 elements of the block
                idxs = [b_idx*2, b_idx*2+1, b_idx*2+4, b_idx*2+5]
                for k in range(4):
                    full_u[idxs[k]] = coeff_out * u_in[k]
            
            full_v = [0]*16
            for b_idx in range(4):
                coeff_out = v_out_blocks[b_idx]
                if coeff_out == 0: continue
                idxs = [b_idx*2, b_idx*2+1, b_idx*2+4, b_idx*2+5]
                for k in range(4):
                    full_v[idxs[k]] = coeff_out * v_in[k]
            
            # w_out_blocks is length 4 (coeffs for C00, C01, C10, C11 blocks)
            # w_in is length 4 (coeffs for local C elements)
            # Full W vector is length 16.
            full_w = [0]*16
            for b_idx in range(4):
                coeff_out = w_out_blocks[b_idx]
                if coeff_out == 0: continue
                idxs = [b_idx*2, b_idx*2+1, b_idx*2+4, b_idx*2+5]
                for k in range(4):
                    full_w[idxs[k]] = coeff_out * w_in[k]
            
            add_term(full_u, full_v, full_w)

    # --- Part 2: 4x4x1 decomposition (Rank 16) ---
    # C_col4 = A * B_col4
    # C[i, 4] = sum_j A[i, j] B[j, 4]
    # W index: 4*n + i = 16 + i
    # V index: j*p + 4 = 5j + 4
    # U index: i*m + j = 4i + j
    
    for i in range(4):
        for j in range(4):
            u = [0]*16
            u[4*i + j] = 1
            
            v = [0]*20
            v[5*j + 4] = 1
            
            w = [0]*20
            w[16 + i] = 1
            
            add_term(u, v, w)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
