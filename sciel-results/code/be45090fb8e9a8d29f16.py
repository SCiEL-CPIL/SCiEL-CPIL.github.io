"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U_A: 4 entries (A00, A01, A10, A11)
    # V_B: 4 entries (B00, B01, B10, B11)
    # W_C: 4 entries (C00, C01, C10, C11)
    strassen_coeffs = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
    ]
    
    # 8 blocks
    for block_idx in range(8):
        a_idx = block_idx // 4  # 0 or 1, selects column block of A and row block of B
        b_col_idx = block_idx % 4 # 0..3, selects column block of B and column block of C
        
        for t in range(7):
            col = block_idx * 7 + t
            u_coeff, v_coeff, w_coeff = strassen_coeffs[t]
            
            # Fill U
            # A local (r, c) -> global row r, col 2*a_idx + c
            # U index: r*m + c_global
            for r in range(2):
                for c_local in range(2):
                    c_global = 2 * a_idx + c_local
                    u_row = r * m + c_global
                    U[u_row][col] = u_coeff[r * 2 + c_local]
                    
            # Fill V
            # B local (r, c) -> global row 2*a_idx + r, col 2*b_col_idx + c
            # V index: r_global * p + c_global
            for r_local in range(2):
                for c_local in range(2):
                    r_global = 2 * a_idx + r_local
                    c_global = 2 * b_col_idx + c_local
                    v_row = r_global * p + c_global
                    V[v_row][col] = v_coeff[r_local * 2 + c_local]
                    
            # Fill W
            # C local (r, c) -> global row r, col 2*b_col_idx + c
            # W index: c_global * n + r
            for r in range(2):
                for c_local in range(2):
                    c_global = 2 * b_col_idx + c_local
                    w_row = c_global * n + r
                    W[w_row][col] = w_coeff[r * 2 + c_local]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
