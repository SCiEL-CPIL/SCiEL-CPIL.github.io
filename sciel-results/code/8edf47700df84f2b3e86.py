"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # We decompose the 2x4x8 multiplication into 8 blocks of 2x2x2.
    # j (dim 4) is split into 2 groups of 2.
    # k (dim 8) is split into 4 groups of 2.
    # i (dim 2) is not split, it's the full dimension for each block.
    # Each 2x2x2 block uses Strassen rank 7.
    # Total rank = 2 * 4 * 7 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # (8, 56)
    V = [[0] * rank for _ in range(m * p)]  # (32, 56)
    W = [[0] * rank for _ in range(p * n)]  # (16, 56)
    
    # Strassen 2x2 * 2x2 coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # u_coeffs[m_idx][(i_loc, j_loc)]
    u_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): 1},       # M2
        {(0,0): 1},                  # M3
        {(1,1): 1},                  # M4
        {(0,0): 1, (0,1): 1},       # M5
        {(1,0): 1, (0,0): -1},      # M6
        {(0,1): 1, (1,1): -1}       # M7
    ]
    
    # v_coeffs[m_idx][(j_loc, k_loc)]
    v_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,0): 1},                  # M2
        {(0,1): 1, (1,1): -1},      # M3
        {(1,0): 1, (0,0): -1},      # M4
        {(1,1): 1},                  # M5
        {(0,0): 1, (0,1): 1},       # M6
        {(1,0): 1, (1,1): 1}        # M7
    ]
    
    # w_coeffs[m_idx][(k_loc, i_loc)]
    # Note: W is indexed by [k*n + i]
    w_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1 -> C00, C11
        {(1,0): 1, (1,1): -1},      # M2 -> C10, C11
        {(0,1): 1, (1,1): 1},       # M3 -> C01, C11
        {(0,0): 1, (1,0): 1},       # M4 -> C00, C10
        {(0,0): -1, (0,1): 1},      # M5 -> C00, C01
        {(1,1): 1},                  # M6 -> C11
        {(0,0): 1}                   # M7 -> C00
    ]
    
    col = 0
    for jg in range(2):
        j_offset = jg * 2
        for kg in range(4):
            k_offset = kg * 2
            for m_idx in range(7):
                # Fill U
                for (i_loc, j_loc), c_u in u_coeffs[m_idx].items():
                    i_global = i_loc
                    j_global = j_offset + j_loc
                    U[i_global * m + j_global][col] += c_u
                
                # Fill V
                for (j_loc, k_loc), c_v in v_coeffs[m_idx].items():
                    j_global = j_offset + j_loc
                    k_global = k_offset + k_loc
                    V[j_global * p + k_global][col] += c_v
                
                # Fill W
                for (k_loc, i_loc), c_w in w_coeffs[m_idx].items():
                    k_global = k_offset + k_loc
                    i_global = i_loc
                    W[k_global * n + i_global][col] += c_w
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
