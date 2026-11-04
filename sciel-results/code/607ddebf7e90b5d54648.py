"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=3, m=4, p=8
    # Rank calculation:
    # Split m=4 into two 2s.
    # Split p=8 into four 2s.
    # For each (m_half, p_quarter) pair:
    #   Split n=3 into 2 and 1.
    #   2x2x2 part: Strassen (rank 7)
    #   1x2x2 part: Schoolbook (rank 4)
    #   Total per pair: 11
    # Number of pairs: 2 (m_h) * 4 (p_q) = 8
    # Total rank: 8 * 11 = 88
    
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    # u_coords: list of (i, l) with coefficients
    # v_coords: list of (l, j) with coefficients
    # w_coords: list of (j, i) with coefficients
    def add_term(u_coords, v_coords, w_coords):
        nonlocal col
        for (i, l), coeff in u_coords:
            U[i * m + l][col] += coeff
        for (l, j), coeff in v_coords:
            V[l * p + j][col] += coeff
        for (j, i), coeff in w_coords:
            W[j * n + i][col] += coeff
        col += 1
        
    # Strassen terms for 2x2x2
    # Local indices 0,1 for n, m, p
    # M1: (A00+A11)(B00+B11) -> C00(+1), C11(+1)
    strassen_terms = [
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # M2: (A10+A11)B00 -> C10(+1)
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1)]),
        # M3: A00(B01-B11) -> C01(+1)
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1)]),
        # M4: A11(B10-B00) -> C10(+1), C00(+1)
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # M5: (A00+A01)B11 -> C00(-1), C01(+1)
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (1,0,1)]),
        # M6: (A10-A00)(B00+B01) -> C11(+1)
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # M7: (A01-A11)(B10+B11) -> C00(+1), C11(-1)
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1), (1,1,-1)])
    ]
    
    # Schoolbook for 1x2x2
    # A is 1x2, B is 2x2. C is 1x2.
    # C0 = A0B00 + A1B10
    # C1 = A0B01 + A1B11
    schoolbook_1x2x2 = [
        # Term 1: A0 * B00 -> C0
        ([(0,0,1)], [(0,0,1)], [(0,0,1)]),
        # Term 2: A1 * B10 -> C0
        ([(0,1,1)], [(1,0,1)], [(0,0,1)]),
        # Term 3: A0 * B01 -> C1
        ([(0,0,1)], [(0,1,1)], [(1,0,1)]),
        # Term 4: A1 * B11 -> C1
        ([(0,1,1)], [(1,1,1)], [(1,0,1)])
    ]
    
    # Iterate over m halves and p quarters
    for m_half in range(2):
        l_offset = m_half * 2
        for p_quarter in range(4):
            j_offset = p_quarter * 2
            
            # Add Strassen terms for n_half=0 (rows 0,1)
            # Local n indices 0,1 map to global 0,1
            # Local m indices 0,1 map to global l_offset+0, l_offset+1
            # Local p indices 0,1 map to global j_offset+0, j_offset+1
            
            for u_c, v_c, w_c in strassen_terms:
                # u_c is list of (local_i, local_l, coeff)
                u_g = [((0 * 2 + li), (l_offset + ll), coeff) for (li, ll, coeff) in u_c]
                v_g = [((l_offset + ll), (j_offset + lj), coeff) for (ll, lj, coeff) in v_c]
                w_g = [((j_offset + lj), (0 * 2 + li), coeff) for (lj, li, coeff) in w_c]
                
                # Note: Strassen terms use local n indices 0,1.
                # Global n indices for this block are 0,1.
                
                add_term(u_g, v_g, w_g)
                
            # Add Schoolbook terms for n_half=1 (row 2)
            # Local n index 0 maps to global 2
            # Local m indices 0,1 map to global l_offset+0, l_offset+1
            # Local p indices 0,1 map to global j_offset+0, j_offset+1
            
            for u_c, v_c, w_c in schoolbook_1x2x2:
                # u_c is list of (local_i, local_l, coeff)
                # local_i is always 0 for 1x2x2, but represents the single row.
                # So global i is 2.
                u_g = [((2), (l_offset + ll), coeff) for (li, ll, coeff) in u_c]
                v_g = [((l_offset + ll), (j_offset + lj), coeff) for (ll, lj, coeff) in v_c]
                w_g = [((j_offset + lj), (2), coeff) for (lj, li, coeff) in w_c]
                
                add_term(u_g, v_g, w_g)
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
