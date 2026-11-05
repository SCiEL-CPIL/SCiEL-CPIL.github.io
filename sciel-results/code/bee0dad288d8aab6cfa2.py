"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a term to the decomposition
    # u_idx: index in U (i*m + j)
    # v_idx: index in V (j*p + k)
    # w_idx: index in W (k*n + i)
    # col: current rank column
    # coeffs: (coeff_u, coeff_v, coeff_w)
    def add_term(u_idx, v_idx, w_idx, col, coeff_u, coeff_v, coeff_w):
        U[u_idx][col] += coeff_u
        V[v_idx][col] += coeff_v
        W[w_idx][col] += coeff_w

    # Helper for Strassen 2x2x2
    # A is 2x2, B is 2x2, C is 2x2
    # u_base: (i, j) base for A. A[0,0] is at u_base[0]*m + u_base[1]
    # v_base: (j, k) base for B. B[0,0] is at v_base[0]*p + v_base[1]
    # w_base: (k, i) base for C. C[0,0] is at w_base[0]*n + w_base[1]
    def add_strassen(u_base, v_base, w_base, col_start):
        # u_base = (i, j), v_base = (j, k), w_base = (k, i)
        # A indices: (i, j), (i, j+1), (i+1, j), (i+1, j+1)
        # B indices: (j, k), (j, k+1), (j+1, k), (j+1, k+1)
        # C indices: (i, k), (i, k+1), (i+1, k), (i+1, k+1)
        
        i, j = u_base
        jv, k = v_base
        kw, iw = w_base
        
        # A elements
        u00 = i * m + j
        u01 = i * m + (j + 1)
        u10 = (i + 1) * m + j
        u11 = (i + 1) * m + (j + 1)
        
        # B elements
        v00 = jv * p + k
        v01 = jv * p + (k + 1)
        v10 = (jv + 1) * p + k
        v11 = (jv + 1) * p + (k + 1)
        
        # C elements
        w00 = kw * n + iw
        w01 = kw * n + (iw + 1)
        w10 = (kw + 1) * n + iw
        w11 = (kw + 1) * n + (iw + 1)
        
        # M1 = (A00 + A11) * (B00 + B11)
        add_term(u00, v00, w00, col_start, 1, 1, 1)
        add_term(u00, v00, w01, col_start, 1, 1, 1)
        add_term(u00, v00, w10, col_start, 1, 1, 1)
        add_term(u00, v00, w11, col_start, 1, 1, 1)
        add_term(u11, v00, w00, col_start, 1, 1, 1)
        add_term(u11, v00, w01, col_start, 1, 1, 1)
        add_term(u11, v00, w10, col_start, 1, 1, 1)
        add_term(u11, v00, w11, col_start, 1, 1, 1)
        add_term(u00, v11, w00, col_start, 1, 1, 1)
        add_term(u00, v11, w01, col_start, 1, 1, 1)
        add_term(u00, v11, w10, col_start, 1, 1, 1)
        add_term(u00, v11, w11, col_start, 1, 1, 1)
        add_term(u11, v11, w00, col_start, 1, 1, 1)
        add_term(u11, v11, w01, col_start, 1, 1, 1)
        add_term(u11, v11, w10, col_start, 1, 1, 1)
        add_term(u11, v11, w11, col_start, 1, 1, 1)
        
        # M2 = (A10 + A11) * B00
        add_term(u10, v00, w00, col_start + 1, 1, 1, 1)
        add_term(u10, v00, w01, col_start + 1, 1, 1, 1)
        add_term(u10, v00, w10, col_start + 1, 1, 1, 1)
        add_term(u10, v00, w11, col_start + 1, 1, 1, 1)
        add_term(u11, v00, w00, col_start + 1, 1, 1, 1)
        add_term(u11, v00, w01, col_start + 1, 1, 1, 1)
        add_term(u11, v00, w10, col_start + 1, 1, 1, 1)
        add_term(u11, v00, w11, col_start + 1, 1, 1, 1)
        
        # M3 = A00 * (B01 - B11)
        add_term(u00, v01, w00, col_start + 2, 1, 1, 1)
        add_term(u00, v01, w01, col_start + 2, 1, 1, 1)
        add_term(u00, v01, w10, col_start + 2, 1, 1, 1)
        add_term(u00, v01, w11, col_start + 2, 1, 1, 1)
        add_term(u00, v11, w00, col_start + 2, 1, -1, 1)
        add_term(u00, v11, w01, col_start + 2, 1, -1, 1)
        add_term(u00, v11, w10, col_start + 2, 1, -1, 1)
        add_term(u00, v11, w11, col_start + 2, 1, -1, 1)
        
        # M4 = A11 * (B10 - B00)
        add_term(u11, v10, w00, col_start + 3, 1, 1, 1)
        add_term(u11, v10, w01, col_start + 3, 1, 1, 1)
        add_term(u11, v10, w10, col_start + 3, 1, 1, 1)
        add_term(u11, v10, w11, col_start + 3, 1, 1, 1)
        add_term(u11, v00, w00, col_start + 3, 1, -1, 1)
        add_term(u11, v00, w01, col_start + 3, 1, -1, 1)
        add_term(u11, v00, w10, col_start + 3, 1, -1, 1)
        add_term(u11, v00, w11, col_start + 3, 1, -1, 1)
        
        # M5 = (A00 + A01) * B11
        add_term(u00, v11, w00, col_start + 4, 1, 1, 1)
        add_term(u00, v11, w01, col_start + 4, 1, 1, 1)
        add_term(u00, v11, w10, col_start + 4, 1, 1, 1)
        add_term(u00, v11, w11, col_start + 4, 1, 1, 1)
        add_term(u01, v11, w00, col_start + 4, 1, 1, 1)
        add_term(u01, v11, w01, col_start + 4, 1, 1, 1)
        add_term(u01, v11, w10, col_start + 4, 1, 1, 1)
        add_term(u01, v11, w11, col_start + 4, 1, 1, 1)
        
        # M6 = (A10 - A00) * (B00 + B01)
        add_term(u10, v00, w00, col_start + 5, 1, 1, 1)
        add_term(u10, v00, w01, col_start + 5, 1, 1, 1)
        add_term(u10, v00, w10, col_start + 5, 1, 1, 1)
        add_term(u10, v00, w11, col_start + 5, 1, 1, 1)
        add_term(u10, v01, w00, col_start + 5, 1, 1, 1)
        add_term(u10, v01, w01, col_start + 5, 1, 1, 1)
        add_term(u10, v01, w10, col_start + 5, 1, 1, 1)
        add_term(u10, v01, w11, col_start + 5, 1, 1, 1)
        add_term(u00, v00, w00, col_start + 5, -1, 1, 1)
        add_term(u00, v00, w01, col_start + 5, -1, 1, 1)
        add_term(u00, v00, w10, col_start + 5, -1, 1, 1)
        add_term(u00, v00, w11, col_start + 5, -1, 1, 1)
        add_term(u00, v01, w00, col_start + 5, -1, 1, 1)
        add_term(u00, v01, w01, col_start + 5, -1, 1, 1)
        add_term(u00, v01, w10, col_start + 5, -1, 1, 1)
        add_term(u00, v01, w11, col_start + 5, -1, 1, 1)
        
        # M7 = (A01 - A11) * (B10 + B11)
        add_term(u01, v10, w00, col_start + 6, 1, 1, 1)
        add_term(u01, v10, w01, col_start + 6, 1, 1, 1)
        add_term(u01, v10, w10, col_start + 6, 1, 1, 1)
        add_term(u01, v10, w11, col_start + 6, 1, 1, 1)
        add_term(u01, v11, w00, col_start + 6, 1, 1, 1)
        add_term(u01, v11, w01, col_start + 6, 1, 1, 1)
        add_term(u01, v11, w10, col_start + 6, 1, 1, 1)
        add_term(u01, v11, w11, col_start + 6, 1, 1, 1)
        add_term(u11, v10, w00, col_start + 6, -1, 1, 1)
        add_term(u11, v10, w01, col_start + 6, -1, 1, 1)
        add_term(u11, v10, w10, col_start + 6, -1, 1, 1)
        add_term(u11, v10, w11, col_start + 6, -1, 1, 1)
        add_term(u11, v11, w00, col_start + 6, -1, 1, 1)
        add_term(u11, v11, w01, col_start + 6, -1, 1, 1)
        add_term(u11, v11, w10, col_start + 6, -1, 1, 1)
        add_term(u11, v11, w11, col_start + 6, -1, 1, 1)
        
        return col_start + 7

    # Helper for Schoolbook 2x2x1
    # A is 2x2, B is 2x1, C is 2x1
    def add_schoolbook_221(u_base, v_base, w_base, col_start):
        i, j = u_base
        jv, k = v_base
        kw, iw = w_base
        
        u00 = i * m + j
        u01 = i * m + (j + 1)
        u10 = (i + 1) * m + j
        u11 = (i + 1) * m + (j + 1)
        
        v00 = jv * p + k
        v10 = (jv + 1) * p + k
        
        w00 = kw * n + iw
        w10 = (kw + 1) * n + iw
        
        # C0 = A00*B00 + A01*B10
        add_term(u00, v00, w00, col_start, 1, 1, 1)
        add_term(u01, v10, w00, col_start, 1, 1, 1)
        
        # C1 = A10*B00 + A11*B10
        add_term(u10, v00, w10, col_start + 1, 1, 1, 1)
        add_term(u11, v10, w10, col_start + 1, 1, 1, 1)
        
        return col_start + 2

    # Helper for Schoolbook 1x2x2
    # A is 1x2, B is 2x2, C is 1x2
    def add_schoolbook_122(u_base, v_base, w_base, col_start):
        i, j = u_base
        jv, k = v_base
        kw, iw = w_base
        
        u00 = i * m + j
        u01 = i * m + (j + 1)
        
        v00 = jv * p + k
        v01 = jv * p + (k + 1)
        v10 = (jv + 1) * p + k
        v11 = (jv + 1) * p + (k + 1)
        
        w00 = kw * n + iw
        w01 = kw * n + (iw + 1)
        
        # C0 = A0*B00 + A1*B10
        add_term(u00, v00, w00, col_start, 1, 1, 1)
        add_term(u01, v10, w00, col_start, 1, 1, 1)
        
        # C1 = A0*B01 + A1*B11
        add_term(u00, v01, w01, col_start + 1, 1, 1, 1)
        add_term(u01, v11, w01, col_start + 1, 1, 1, 1)
        
        return col_start + 2

    # Helper for Schoolbook 1x2x1
    # A is 1x2, B is 2x1, C is 1x1
    def add_schoolbook_121(u_base, v_base, w_base, col_start):
        i, j = u_base
        jv, k = v_base
        kw, iw = w_base
        
        u00 = i * m + j
        u01 = i * m + (j + 1)
        
        v00 = jv * p + k
        v10 = (jv + 1) * p + k
        
        w00 = kw * n + iw
        
        # C0 = A0*B0 + A1*B1
        add_term(u00, v00, w00, col_start, 1, 1, 1)
        add_term(u01, v10, w00, col_start, 1, 1, 1)
        
        return col_start + 2

    col = 0
    
    # Part 1: Top 2 rows of A (i=0,1)
    # A_L: i in {0,1}, j in {0,1}
    # A_R: i in {0,1}, j in {2,3}
    # B_L: j in {0,1}, k in {0,1,2,3,4}
    # B_R: j in {2,3}, k in {0,1,2,3,4}
    
    # 1.1: A_L (2x2) * B_L (2x5)
    # Split B_L into B_L1 (2x2, k=0,1), B_L2 (2x2, k=2,3), B_L3 (2x1, k=4)
    
    # A_L * B_L1 (2x2x2) -> C (2x2) at i=0,1; k=0,1
    col = add_strassen((0, 0), (0, 0), (0, 0), col)
    
    # A_L * B_L2 (2x2x2) -> C (2x2) at i=0,1; k=2,3
    col = add_strassen((0, 0), (0, 2), (2, 0), col)
    
    # A_L * B_L3 (2x2x1) -> C (2x1) at i=0,1; k=4
    col = add_schoolbook_221((0, 0), (0, 4), (4, 0), col)
    
    # 1.2: A_R (2x2) * B_R (2x5)
    # Split B_R into B_R1 (2x2, k=0,1), B_R2 (2x2, k=2,3), B_R3 (2x1, k=4)
    
    # A_R * B_R1 (2x2x2) -> C (2x2) at i=0,1; k=0,1
    col = add_strassen((0, 2), (2, 0), (0, 0), col)
    
    # A_R * B_R2 (2x2x2) -> C (2x2) at i=0,1; k=2,3
    col = add_strassen((0, 2), (2, 2), (2, 0), col)
    
    # A_R * B_R3 (2x2x1) -> C (2x1) at i=0,1; k=4
    col = add_schoolbook_221((0, 2), (2, 4), (4, 0), col)
    
    # Part 2: Bottom 1 row of A (i=2)
    # A_bL: i=2, j in {0,1}
    # A_bR: i=2, j in {2,3}
    
    # 2.1: A_bL (1x2) * B_L (2x5)
    # Split B_L into B_L1 (2x2, k=0,1), B_L2 (2x2, k=2,3), B_L3 (2x1, k=4)
    
    # A_bL * B_L1 (1x2x2) -> C (1x2) at i=2; k=0,1
    col = add_schoolbook_122((2, 0), (0, 0), (0, 2), col)
    
    # A_bL * B_L2 (1x2x2) -> C (1x2) at i=2; k=2,3
    col = add_schoolbook_122((2, 0), (0, 2), (2, 2), col)
    
    # A_bL * B_L3 (1x2x1) -> C (1x1) at i=2; k=4
    col = add_schoolbook_121((2, 0), (0, 4), (4, 2), col)
    
    # 2.2: A_bR (1x2) * B_R (2x5)
    # Split B_R into B_R1 (2x2, k=0,1), B_R2 (2x2, k=2,3), B_R3 (2x1, k=4)
    
    # A_bR * B_R1 (1x2x2) -> C (1x2) at i=2; k=0,1
    col = add_schoolbook_122((2, 2), (2, 0), (0, 2), col)
    
    # A_bR * B_R2 (1x2x2) -> C (1x2) at i=2; k=2,3
    col = add_schoolbook_122((2, 2), (2, 2), (2, 2), col)
    
    # A_bR * B_R3 (1x2x1) -> C (1x1) at i=2; k=4
    col = add_schoolbook_121((2, 2), (2, 4), (4, 2), col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
