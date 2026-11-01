"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    # Block Strassen decomposition of rank 21 for 2x2 * 2x6
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 * 2x2
    # Terms: (u_coeffs, v_coeffs, w_coeffs)
    # u: [a, b, c, d] corresponding to A[0,0], A[0,1], A[1,0], A[1,1]
    # v: [b00, b01, b10, b11] corresponding to B[0,0], B[0,1], B[1,0], B[1,1]
    # w: [c00, c01, c10, c11] corresponding to C[0,0], C[0,1], C[1,0], C[1,1]
    
    strassen_terms = [
        # M1: (a+d)(b00+b11) -> C00 += 1, C11 += 1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2: (c+d)b01 -> C01 += 1, C11 -= 1
        ([0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, -1]),
        # M3: a(b10-b11) -> C10 += 1, C11 += 1
        ([1, 0, 0, 0], [0, 0, 1, -1], [0, 0, 1, 1]),
        # M4: d(b00-b10) -> C00 += 1, C10 += 1
        ([0, 0, 0, 1], [1, 0, -1, 0], [1, 0, 1, 0]),
        # M5: (a+b)b11 -> C00 -= 1, C01 += 1
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6: (c-d)(b00+b01) -> C11 += 1
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # M7: (b-d)(b10+b11) -> C00 += 1
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    # Split B into 3 blocks of 2 columns each
    for block in range(3):
        # Global column offset for this block
        col_offset = block * 2
        
        for t_idx in range(7):
            r = block * 7 + t_idx
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t_idx]
            
            # Update U
            # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
            # U index: i*m + j
            for idx, val in enumerate(u_coeffs):
                if val != 0:
                    i = idx // 2
                    j = idx % 2
                    U[i * m + j][r] += val
                    
            # Update V
            # B local indices: 00, 01, 10, 11
            # Global B indices: j, col_offset + k_local
            # V index: j*p + k_global
            for idx, val in enumerate(v_coeffs):
                if val != 0:
                    j = idx // 2
                    k_local = idx % 2
                    k_global = col_offset + k_local
                    V[j * p + k_global][r] += val
                    
            # Update W
            # C local indices: 00, 01, 10, 11
            # Global C indices: i, col_offset + k_local
            # W index: k_global*n + i
            for idx, val in enumerate(w_coeffs):
                if val != 0:
                    i = idx // 2
                    k_local = idx % 2
                    k_global = col_offset + k_local
                    W[k_global * n + i][r] += val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
