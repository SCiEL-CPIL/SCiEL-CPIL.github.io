"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 Strassen terms = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2:
    # terms[s] = (U_coeffs, V_coeffs, W_coeffs)
    # Local 2x2 index mapping: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    terms = [
        # M1 = (x00+x11)(y00+y11) -> z00 += M1, z11 += M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (x10+x11)y00 -> z10 += M2, z11 -= M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = x00(y01-y11) -> z01 += M3, z11 += M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = x11(y10-y00) -> z00 += M4, z10 += M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (x00+x01)y11 -> z00 -= M5, z01 += M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (x10-x00)(y00+y01) -> z11 += M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (x01-x11)(y10+y11) -> z00 += M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Block definitions for 2x4 x 4x4 -> 2x4:
    # Block b: A uses cols [2*(b%2):2*(b%2)+2], B uses rows [2*(b%2):2*(b%2)+2] cols [2*(b//2):2*(b//2)+2]
    # C uses cols [2*(b//2):2*(b//2)+2]
    
    # Global index mappings:
    # A[i,j]: global = i*4 + j
    # B[j,r]: global = j*4 + r  
    # C[i,r]: global = r*2 + i (trace convention)
    
    # For block b:
    #   a_col_offset = 2*(b % 2)
    #   b_row_offset = 2*(b % 2)
    #   b_col_offset = 2*(b // 2)
    #   c_col_offset = 2*(b // 2)
    
    for b in range(4):
        a_col_off = 2 * (b % 2)
        b_row_off = 2 * (b % 2)
        b_col_off = 2 * (b // 2)
        c_col_off = 2 * (b // 2)
        
        for s in range(7):
            col = b * 7 + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]
            
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                i = li
                j = a_col_off + lj
                g_idx = i * 4 + j
                U[g_idx][col] += coeff
                
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                j = b_row_off + li
                r = b_col_off + lj
                g_idx = j * 4 + r
                V[g_idx][col] += coeff
                
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                i = li
                r = c_col_off + lj
                g_idx = r * 2 + i
                W[g_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
