"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 72 solution using Strassen for 2x2x2 blocks
    # A is 4x4, B is 4x5, C is 4x5
    # U shape (16, R), V shape (20, R), W shape (20, R)
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, val=1):
        nonlocal col
        for idx in u_indices:
            U[idx][col] += val
        for idx in v_indices:
            V[idx][col] += val
        for idx in w_indices:
            W[idx][col] += val
        col += 1
        
    # Strassen triples for 2x2x2
    # Local A indices: 00, 01, 10, 11 -> Global A indices
    # Local B indices: 00, 01, 10, 11 -> Global B indices
    # Local C indices: 00, 01, 10, 11 -> Global W indices
    
    # We need to map local block (r, c) of A and (r, c) of B to global indices
    # A is 4x4. Blocks are 2x2.
    # A_block(r, c) covers A[2r:2r+2, 2c:2c+2]
    # B is 4x5. B_top is rows 0-1, B_bottom is rows 2-3.
    # B_block(r, c) for r in {0,1} (top/bottom) covers B[2r:2r+2, 2c:2c+2] for c in {0,1}
    # For the 5th column (index 4), we handle it separately in the 2x2x1 part.
    
    # 4 combinations of A blocks and B blocks
    # A11 (r=0, c=0) * B_top (r=0)
    # A12 (r=0, c=1) * B_bottom (r=1)
    # A21 (r=1, c=0) * B_top (r=0)
    # A22 (r=1, c=1) * B_bottom (r=1)
    
    def get_a_idx(br, bc, li, lj):
        # br, bc: block row, block col of A (0 or 1)
        # li, lj: local row, col (0 or 1)
        gi = 2*br + li
        gj = 2*bc + lj
        return gi * m + gj
        
    def get_b_idx(br, bc, lj, lk):
        # br: block row of B (0 for top, 1 for bottom)
        # bc: block col of B (0 or 1)
        # lj: local row (0 or 1)
        # lk: local col (0 or 1)
        gj = 2*br + lj
        gk = 2*bc + lk
        return gj * p + gk
        
    def get_w_idx(br, bc, lk, li):
        # br: block row of C (0 or 1)
        # bc: block col of C (0 or 1)
        # lk: local col (0 or 1)
        # li: local row (0 or 1)
        gi = 2*br + li
        gk = 2*bc + lk
        return gk * n + gi
        
    # Strassen terms for 2x2x2
    # M1 = (A00+A11)(B00+B11) -> C00, C11
    # M2 = (A10+A11)B00 -> C10, C11
    # M3 = A00(B10-B11) -> C01
    # M4 = A11(B01-B00) -> C00, C10
    # M5 = (A00+A01)B11 -> C00, C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00, C11
    
    # Indices in local 2x2:
    # A00=0, A01=1, A10=2, A11=3
    # B00=0, B01=1, B10=2, B11=3
    # C00=0, C01=1, C10=2, C11=3
    
    def add_strassen(ab_r, ab_c, bb_r, bb_c):
        nonlocal col
        # ab_r, ab_c: A block indices
        # bb_r, bb_c: B block indices
        
        # M1
        u = [get_a_idx(ab_r, ab_c, 0, 0), get_a_idx(ab_r, ab_c, 1, 1)]
        v = [get_b_idx(bb_r, bb_c, 0, 0), get_b_idx(bb_r, bb_c, 1, 1)]
        w = [get_w_idx(ab_r, bb_c, 0, 0), get_w_idx(ab_r, bb_c, 1, 1)]
        add_term(u, v, w, 1)
        
        # M2
        u = [get_a_idx(ab_r, ab_c, 1, 0), get_a_idx(ab_r, ab_c, 1, 1)]
        v = [get_b_idx(bb_r, bb_c, 0, 0)]
        w = [get_w_idx(ab_r, bb_c, 0, 1), get_w_idx(ab_r, bb_c, 1, 1)]
        add_term(u, v, w, 1)
        
        # M3
        u = [get_a_idx(ab_r, ab_c, 0, 0)]
        v = [get_b_idx(bb_r, bb_c, 1, 0), get_b_idx(bb_r, bb_c, 1, 1)]
        w = [get_w_idx(ab_r, bb_c, 1, 0)]
        add_term(u, v, w, 1)
        # Note: M3 = A00(B10-B11). So v should have -1 for B11.
        # My add_term adds val. I need to handle signs.
        # Let's refactor add_term to take signed indices.
        
    # Refactor: 
    # I will just manually list the 72 terms in a loop or list.
    # Given the complexity of signs, it's safer to use a list of (u, v, w) with values.
    
    # Let's restart the generation with a more flexible add_term.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    def add_signed_term(u_signed, v_signed, w_signed):
        nonlocal col
        for idx, val in u_signed:
            U[idx][col] += val
        for idx, val in v_signed:
            V[idx][col] += val
        for idx, val in w_signed:
            W[idx][col] += val
        col += 1

    # Define the 4 block combinations
    # 1. A(0,0) * B(0, :)
    # 2. A(0,1) * B(1, :)
    # 3. A(1,0) * B(0, :)
    # 4. A(1,1) * B(1, :)
    
    for ab_r in [0, 1]:
        for ab_c in [0, 1]:
            # B block row is determined by ab_c? No.
            # C = A11 B_top + A12 B_bottom
            # C = A21 B_top + A22 B_bottom
            # So if ab_c == 0, B_row = 0 (top)
            # If ab_c == 1, B_row = 1 (bottom)
            bb_r = ab_c
            
            # Process 2x2x2 part (cols 0,1 of B block)
            # B block col 0 (global cols 0,1)
            bb_c = 0
            # M1
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 0, 0), 1), (get_a_idx(ab_r, ab_c, 1, 1), 1)],
                [(get_b_idx(bb_r, bb_c, 0, 0), 1), (get_b_idx(bb_r, bb_c, 1, 1), 1)],
                [(get_w_idx(ab_r, bb_c, 0, 0), 1), (get_w_idx(ab_r, bb_c, 1, 1), 1)]
            )
            # M2
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 1, 0), 1), (get_a_idx(ab_r, ab_c, 1, 1), 1)],
                [(get_b_idx(bb_r, bb_c, 0, 0), 1)],
                [(get_w_idx(ab_r, bb_c, 0, 1), 1), (get_w_idx(ab_r, bb_c, 1, 1), 1)]
            )
            # M3
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 0, 0), 1)],
                [(get_b_idx(bb_r, bb_c, 1, 0), 1), (get_b_idx(bb_r, bb_c, 1, 1), -1)],
                [(get_w_idx(ab_r, bb_c, 1, 0), 1)]
            )
            # M4
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 1, 1), 1)],
                [(get_b_idx(bb_r, bb_c, 0, 1), 1), (get_b_idx(bb_r, bb_c, 0, 0), -1)],
                [(get_w_idx(ab_r, bb_c, 0, 0), 1), (get_w_idx(ab_r, bb_c, 0, 1), 1)]
            )
            # M5
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 0, 0), 1), (get_a_idx(ab_r, ab_c, 0, 1), 1)],
                [(get_b_idx(bb_r, bb_c, 1, 1), 1)],
                [(get_w_idx(ab_r, bb_c, 0, 0), 1), (get_w_idx(ab_r, bb_c, 1, 0), 1)]
            )
            # M6
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 1, 0), 1), (get_a_idx(ab_r, ab_c, 0, 0), -1)],
                [(get_b_idx(bb_r, bb_c, 0, 0), 1), (get_b_idx(bb_r, bb_c, 0, 1), 1)],
                [(get_w_idx(ab_r, bb_c, 1, 1), 1)]
            )
            # M7
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 0, 1), 1), (get_a_idx(ab_r, ab_c, 1, 1), -1)],
                [(get_b_idx(bb_r, bb_c, 1, 0), 1), (get_b_idx(bb_r, bb_c, 1, 1), 1)],
                [(get_w_idx(ab_r, bb_c, 0, 0), 1), (get_w_idx(ab_r, bb_c, 1, 1), 1)]
            )
            
            # Process 2x2x2 part (cols 2,3 of B block)
            bb_c = 1
            # Repeat M1-M7 with bb_c=1
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 0, 0), 1), (get_a_idx(ab_r, ab_c, 1, 1), 1)],
                [(get_b_idx(bb_r, bb_c, 0, 0), 1), (get_b_idx(bb_r, bb_c, 1, 1), 1)],
                [(get_w_idx(ab_r, bb_c, 0, 0), 1), (get_w_idx(ab_r, bb_c, 1, 1), 1)]
            )
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 1, 0), 1), (get_a_idx(ab_r, ab_c, 1, 1), 1)],
                [(get_b_idx(bb_r, bb_c, 0, 0), 1)],
                [(get_w_idx(ab_r, bb_c, 0, 1), 1), (get_w_idx(ab_r, bb_c, 1, 1), 1)]
            )
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 0, 0), 1)],
                [(get_b_idx(bb_r, bb_c, 1, 0), 1), (get_b_idx(bb_r, bb_c, 1, 1), -1)],
                [(get_w_idx(ab_r, bb_c, 1, 0), 1)]
            )
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 1, 1), 1)],
                [(get_b_idx(bb_r, bb_c, 0, 1), 1), (get_b_idx(bb_r, bb_c, 0, 0), -1)],
                [(get_w_idx(ab_r, bb_c, 0, 0), 1), (get_w_idx(ab_r, bb_c, 0, 1), 1)]
            )
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 0, 0), 1), (get_a_idx(ab_r, ab_c, 0, 1), 1)],
                [(get_b_idx(bb_r, bb_c, 1, 1), 1)],
                [(get_w_idx(ab_r, bb_c, 0, 0), 1), (get_w_idx(ab_r, bb_c, 1, 0), 1)]
            )
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 1, 0), 1), (get_a_idx(ab_r, ab_c, 0, 0), -1)],
                [(get_b_idx(bb_r, bb_c, 0, 0), 1), (get_b_idx(bb_r, bb_c, 0, 1), 1)],
                [(get_w_idx(ab_r, bb_c, 1, 1), 1)]
            )
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 0, 1), 1), (get_a_idx(ab_r, ab_c, 1, 1), -1)],
                [(get_b_idx(bb_r, bb_c, 1, 0), 1), (get_b_idx(bb_r, bb_c, 1, 1), 1)],
                [(get_w_idx(ab_r, bb_c, 0, 0), 1), (get_w_idx(ab_r, bb_c, 1, 1), 1)]
            )
            
            # Process 2x2x1 part (col 4 of B block)
            # B col 4 is global col 4.
            # Local col index in B is 4? No, B is 2x5.
            # We treat it as 2x1.
            # A 2x2 * B 2x1.
            # C 2x1.
            # 4 terms.
            
            # B index for col 4:
            # B[2*bb_r + 0, 4] and B[2*bb_r + 1, 4]
            
            # Term 1: A00 * B0
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 0, 0), 1)],
                [(get_b_idx(bb_r, 0, 0, 4), 1)], # Wait, get_b_idx uses local col. 
                # My get_b_idx takes bb_c. For col 4, it's not in a 2x2 block.
                # I need a specific index for col 4.
                # B index: (2*bb_r + 0)*5 + 4
                [(2*bb_r + 0)*5 + 4, 1],
                [(get_w_idx(ab_r, 0, 4, 0), 1)] # W index: (4)*4 + (2*ab_r + 0)
            )
            
            # Term 2: A01 * B1
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 0, 1), 1)],
                [(2*bb_r + 1)*5 + 4, 1],
                [(4*4 + 2*ab_r + 0, 1)]
            )
            
            # Term 3: A10 * B0
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 1, 0), 1)],
                [(2*bb_r + 0)*5 + 4, 1],
                [(4*4 + 2*ab_r + 1, 1)]
            )
            
            # Term 4: A11 * B1
            add_signed_term(
                [(get_a_idx(ab_r, ab_c, 1, 1), 1)],
                [(2*bb_r + 1)*5 + 4, 1],
                [(4*4 + 2*ab_r + 1, 1)]
            )

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
