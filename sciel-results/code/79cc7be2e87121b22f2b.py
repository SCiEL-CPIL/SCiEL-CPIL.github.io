"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 7
    
    # Dimensions
    dim_A = n * m  # 6
    dim_B = m * p  # 21
    dim_C = p * n  # 14
    
    # We will build the decomposition with rank 39
    # 3 Strassen blocks (2x2x2) for cols 0-1, 2-3, 4-5 of B_top
    # 1 Schoolbook block (2x2x1) for col 6 of B_top
    # 7 Schoolbook blocks (2x1x1) for B_bottom
    
    rank = 39
    U = [[0] * rank for _ in range(dim_A)]
    V = [[0] * rank for _ in range(dim_B)]
    W = [[0] * rank for _ in range(dim_C)]
    
    r = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal r
        U[u_idx][r] = u_val
        V[v_idx][r] = v_val
        W[w_idx][r] = w_val
        r += 1

    def add_strassen_block(k):
        # k is the starting column index (0, 2, 4)
        # A indices: 0, 1, 3, 4
        # B indices: k, k+1 (row 0), 7+k, 7+k+1 (row 1)
        # C indices: 2k, 2k+1, 2k+2, 2k+3
        
        a_idx, b_idx, c_idx, d_idx = 0, 1, 3, 4
        e_idx, f_idx = k, k + 1
        g_idx, h_idx = 7 + k, 7 + k + 1
        c00_idx, c10_idx = 2 * k, 2 * k + 1
        c01_idx, c11_idx = 2 * k + 2, 2 * k + 3
        
        # m1 = (a+d)(e+h)
        # C00 += m1, C10 -= m1, C01 -= m1, C11 += m1
        U[a_idx][r] = 1; U[d_idx][r] = 1
        V[e_idx][r] = 1; V[h_idx][r] = 1
        W[c00_idx][r] = 1; W[c10_idx][r] = -1
        W[c01_idx][r] = -1; W[c11_idx][r] = 1
        r += 1
        
        # m2 = (c+d)e
        # C10 += m2, C11 -= m2
        U[c_idx][r] = 1; U[d_idx][r] = 1
        V[e_idx][r] = 1
        W[c10_idx][r] = 1; W[c11_idx][r] = -1
        r += 1
        
        # m3 = a(f-h)
        # C01 += m3, C11 += m3
        U[a_idx][r] = 1
        V[f_idx][r] = 1; V[h_idx][r] = -1
        W[c01_idx][r] = 1; W[c11_idx][r] = 1
        r += 1
        
        # m4 = d(g-e)
        # C00 += m4, C10 += m4
        U[d_idx][r] = 1
        V[g_idx][r] = 1; V[e_idx][r] = -1
        W[c00_idx][r] = 1; W[c10_idx][r] = 1
        r += 1
        
        # m5 = (a+b)h
        # C00 -= m5, C01 += m5
        U[a_idx][r] = 1; U[b_idx][r] = 1
        V[h_idx][r] = 1
        W[c00_idx][r] = -1; W[c01_idx][r] = 1
        r += 1
        
        # m6 = (c-a)(e+f)
        # C11 += m6
        U[c_idx][r] = 1; U[a_idx][r] = -1
        V[e_idx][r] = 1; V[f_idx][r] = 1
        W[c11_idx][r] = 1
        r += 1
        
        # m7 = (b-d)(g+h)
        # C00 += m7
        U[b_idx][r] = 1; U[d_idx][r] = -1
        V[g_idx][r] = 1; V[h_idx][r] = 1
        W[c00_idx][r] = 1
        r += 1

    # Add 3 Strassen blocks for columns 0-1, 2-3, 4-5
    for k in [0, 2, 4]:
        add_strassen_block(k)
        
    # Add 1 Schoolbook block for column 6 (2x2x1)
    k = 6
    a_idx, b_idx, c_idx, d_idx = 0, 1, 3, 4
    e_idx = k
    g_idx = 7 + k
    c0_idx = 2 * k
    c1_idx = 2 * k + 1
    
    # a*e -> C0
    U[a_idx][r] = 1
    V[e_idx][r] = 1
    W[c0_idx][r] = 1
    r += 1
    
    # b*g -> C0
    U[b_idx][r] = 1
    V[g_idx][r] = 1
    W[c0_idx][r] = 1
    r += 1
    
    # c*e -> C1
    U[c_idx][r] = 1
    V[e_idx][r] = 1
    W[c1_idx][r] = 1
    r += 1
    
    # d*g -> C1
    U[d_idx][r] = 1
    V[g_idx][r] = 1
    W[c1_idx][r] = 1
    r += 1
    
    # Add 7 blocks for A_right (2x1) * B_bottom (1x7)
    # A indices: 2, 5
    # B indices: 14+k
    # C indices: 2k, 2k+1
    for k in range(7):
        a02_idx = 2
        a12_idx = 5
        b2k_idx = 14 + k
        c0_idx = 2 * k
        c1_idx = 2 * k + 1
        
        # a02 * b2k -> C0
        U[a02_idx][r] = 1
        V[b2k_idx][r] = 1
        W[c0_idx][r] = 1
        r += 1
        
        # a12 * b2k -> C1
        U[a12_idx][r] = 1
        V[b2k_idx][r] = 1
        W[c1_idx][r] = 1
        r += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
