"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a 2x2x2 Strassen block
    def add_strassen_block(u_a, u_b, u_c, u_d, v_e, v_f, v_g, v_h, w_c00, w_c01, w_c10, w_c11, col_start):
        # M1 = (a+d)(e+h)
        U[u_a][col_start] += 1
        U[u_d][col_start] += 1
        V[v_e][col_start] += 1
        V[v_h][col_start] += 1
        W[w_c00][col_start] += 1
        W[w_c11][col_start] += 1
        
        # M2 = (c+d)(e+h)
        U[u_c][col_start+1] += 1
        U[u_d][col_start+1] += 1
        V[v_e][col_start+1] += 1
        V[v_h][col_start+1] += 1
        W[w_c10][col_start+1] += 1
        W[w_c11][col_start+1] -= 1
        
        # M3 = (a-b)(f-h)
        U[u_a][col_start+2] += 1
        U[u_b][col_start+2] -= 1
        V[v_f][col_start+2] += 1
        V[v_h][col_start+2] -= 1
        W[w_c01][col_start+2] += 1
        W[w_c11][col_start+2] += 1
        
        # M4 = d(g-e)
        U[u_d][col_start+3] += 1
        V[v_g][col_start+3] += 1
        V[v_e][col_start+3] -= 1
        W[w_c00][col_start+3] += 1
        W[w_c10][col_start+3] += 1
        
        # M5 = a(h-g)
        U[u_a][col_start+4] += 1
        V[v_h][col_start+4] += 1
        V[v_g][col_start+4] -= 1
        W[w_c00][col_start+4] -= 1
        W[w_c01][col_start+4] += 1
        
        # M6 = (a+c)(e-f)
        U[u_a][col_start+5] += 1
        U[u_c][col_start+5] += 1
        V[v_e][col_start+5] += 1
        V[v_f][col_start+5] -= 1
        W[w_c11][col_start+5] += 1
        
        # M7 = (b+d)(f-h)
        U[u_b][col_start+6] += 1
        U[u_d][col_start+6] += 1
        V[v_f][col_start+6] += 1
        V[v_h][col_start+6] -= 1
        W[w_c00][col_start+6] -= 1
        
    # Helper to add a 2x2x1 schoolbook block
    def add_schoolbook_221(u_a, u_b, u_c, u_d, v_e, v_g, w_c0, w_c1, col_start):
        # a*e -> c0
        U[u_a][col_start] += 1
        V[v_e][col_start] += 1
        W[w_c0][col_start] += 1
        
        # b*g -> c0
        U[u_b][col_start+1] += 1
        V[v_g][col_start+1] += 1
        W[w_c0][col_start+1] += 1
        
        # c*e -> c1
        U[u_c][col_start+2] += 1
        V[v_e][col_start+2] += 1
        W[w_c1][col_start+2] += 1
        
        # d*g -> c1
        U[u_d][col_start+3] += 1
        V[v_g][col_start+3] += 1
        W[w_c1][col_start+3] += 1

    # Block 1: j in {0, 1}
    # U rows for A[:, 0:2]
    u_a1, u_b1, u_c1, u_d1 = 0, 1, 4, 5
    
    # Sub-block 1a: k in {0, 1}
    # V rows for B[0:2, 0:2]
    v_e1a, v_f1a, v_g1a, v_h1a = 0, 5, 1, 6
    # W rows for C[:, 0:2]
    w_c00_1a, w_c01_1a, w_c10_1a, w_c11_1a = 0, 2, 1, 3
    add_strassen_block(u_a1, u_b1, u_c1, u_d1, v_e1a, v_f1a, v_g1a, v_h1a, 
                       w_c00_1a, w_c01_1a, w_c10_1a, w_c11_1a, 0)
    
    # Sub-block 1b: k in {2, 3}
    # V rows for B[0:2, 2:4]
    v_e1b, v_f1b, v_g1b, v_h1b = 2, 7, 3, 8
    # W rows for C[:, 2:4]
    w_c00_1b, w_c01_1b, w_c10_1b, w_c11_1b = 4, 6, 5, 7
    add_strassen_block(u_a1, u_b1, u_c1, u_d1, v_e1b, v_f1b, v_g1b, v_h1b, 
                       w_c00_1b, w_c01_1b, w_c10_1b, w_c11_1b, 7)
    
    # Sub-block 1c: k in {4}
    # V rows for B[0:2, 4:5]
    v_e1c, v_g1c = 4, 9
    # W rows for C[:, 4:5]
    w_c0_1c, w_c1_1c = 8, 9
    add_schoolbook_221(u_a1, u_b1, u_c1, u_d1, v_e1c, v_g1c, 
                       w_c0_1c, w_c1_1c, 14)

    # Block 2: j in {2, 3}
    # U rows for A[:, 2:4]
    u_a2, u_b2, u_c2, u_d2 = 2, 3, 6, 7
    
    # Sub-block 2a: k in {0, 1}
    # V rows for B[2:4, 0:2]
    v_e2a, v_f2a, v_g2a, v_h2a = 10, 15, 11, 16
    # W rows for C[:, 0:2] (same as block 1)
    add_strassen_block(u_a2, u_b2, u_c2, u_d2, v_e2a, v_f2a, v_g2a, v_h2a, 
                       w_c00_1a, w_c01_1a, w_c10_1a, w_c11_1a, 18)
    
    # Sub-block 2b: k in {2, 3}
    # V rows for B[2:4, 2:4]
    v_e2b, v_f2b, v_g2b, v_h2b = 12, 17, 13, 18
    # W rows for C[:, 2:4] (same as block 1)
    add_strassen_block(u_a2, u_b2, u_c2, u_d2, v_e2b, v_f2b, v_g2b, v_h2b, 
                       w_c00_1b, w_c01_1b, w_c10_1b, w_c11_1b, 25)
    
    # Sub-block 2c: k in {4}
    # V rows for B[2:4, 4:5]
    v_e2c, v_g2c = 14, 19
    # W rows for C[:, 4:5] (same as block 1)
    add_schoolbook_221(u_a2, u_b2, u_c2, u_d2, v_e2c, v_g2c, 
                       w_c0_1c, w_c1_1c, 32)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
