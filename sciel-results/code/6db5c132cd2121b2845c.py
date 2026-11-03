"""Matrix multiplication 2,4,6 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2x2 terms: (U_coeffs, V_coeffs, W_coeffs)
    # Local A indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Local B indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # Local C indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    #
    # P1 = (a+d)(e+h)
    # P2 = (c+d)(e)
    # P3 = a(f-h)
    # P4 = d(g-e)
    # P5 = (a+b)(h)
    # P6 = (c-a)(e+f)
    # P7 = (b-d)(g+h)
    #
    # C00 = P1 - P4 + P6 - P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P3 + P5 - P6
    
    strassen = [
        # P1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # P2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        # P3: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,-1]
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,-1)]),
        # P4: U=[0,0,0,1], V=[0,0,1,-1], W=[-1,0,1,0]
        ([(3,1)], [(2,1),(0,-1)], [(0,-1),(2,1)]),
        # P5: U=[1,1,0,0], V=[0,0,0,1], W=[0,1,0,1]
        ([(0,1),(1,1)], [(3,1)], [(1,1),(3,1)]),
        # P6: U=[-1,0,1,0], V=[1,1,0,0], W=[1,0,0,-1]
        ([(0,-1),(2,1)], [(0,1),(1,1)], [(0,1),(3,-1)]),
        # P7: U=[0,1,0,-1], V=[0,0,1,1], W=[-1,0,0,0]
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,-1)]),
    ]
    
    # Six 2x2x2 blocks:
    # A is 2x4, B is 4x6, C is 2x6
    # Split middle dim (4) into 2+2:
    #   A_left = A[:, 0:2] (2x2), A_right = A[:, 2:4] (2x2)
    #   B_top = B[0:2, :] (2x6), B_bottom = B[2:4, :] (2x6)
    # C = A_left @ B_top + A_right @ B_bottom
    #
    # Each 2x2 @ 2x6 splits into three 2x2 @ 2x2 blocks.
    # Total: 2 (A halves) * 3 (B col groups) = 6 blocks * 7 terms = 42
    
    # Block definitions: (a_col_offset, b_row_offset, c_col_offset)
    # a_col_offset: 0 for A_left, 2 for A_right
    # b_row_offset: 0 for B_top, 2 for B_bottom  
    # c_col_offset: 0, 2, or 4 for the three column groups
    blocks = [
        (0, 0, 0),  # A_left * B_top[:, 0:2]
        (0, 0, 2),  # A_left * B_top[:, 2:4]
        (0, 0, 4),  # A_left * B_top[:, 4:6]
        (2, 2, 0),  # A_right * B_bottom[:, 0:2]
        (2, 2, 2),  # A_right * B_bottom[:, 2:4]
        (2, 2, 4),  # A_right * B_bottom[:, 4:6]
    ]
    
    col = 0
    for (a_col_off, b_row_off, c_col_off) in blocks:
        for (u_coeffs, v_coeffs, w_coeffs) in strassen:
            # Update U: A index = i * m + (a_col_off + j_loc)
            for (loc_idx, coeff) in u_coeffs:
                i_loc, j_loc = divmod(loc_idx, 2)
                u_idx = i_loc * m + (a_col_off + j_loc)
                U[u_idx][col] += coeff
            
            # Update V: B index = (b_row_off + j_loc) * p + (c_col_off + k_loc)
            for (loc_idx, coeff) in v_coeffs:
                j_loc, k_loc = divmod(loc_idx, 2)
                v_idx = (b_row_off + j_loc) * p + (c_col_off + k_loc)
                V[v_idx][col] += coeff
            
            # Update W: C index = (c_col_off + k_loc) * n + i_loc
            for (loc_idx, coeff) in w_coeffs:
                i_loc, k_loc = divmod(loc_idx, 2)
                w_idx = (c_col_off + k_loc) * n + i_loc
                W[w_idx][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
