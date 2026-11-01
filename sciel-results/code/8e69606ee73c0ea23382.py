"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 21 (3 groups of 2x2 Strassen multiplications)
    rank = 21
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]  # 4 x 21
    V = [[0] * rank for _ in range(m * p)]  # 12 x 21
    W = [[0] * rank for _ in range(p * n)]  # 12 x 21
    
    # Strassen coefficients for 2x2 multiplication
    # M is 2x2 (A), N is 2x2 (B_g)
    # P1 = (M00+M11)(N00+N11)
    # P2 = (M10+M11)N00
    # P3 = M00(N10-N11)
    # P4 = M11(N01-N00)
    # P5 = (M00+M01)N11
    # P6 = (M10-M00)(N00+N10)
    # P7 = (M01-M11)(N01+N11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # Indices for U (A): i*m + j
    # A00=0, A01=1, A10=2, A11=3
    
    # Indices for V (B_g): j*p + k
    # For group g, columns are 2g, 2g+1
    # B0,2g = 0*6 + 2g = 2g
    # B0,2g+1 = 0*6 + 2g+1 = 2g+1
    # B1,2g = 1*6 + 2g = 6+2g
    # B1,2g+1 = 1*6 + 2g+1 = 7+2g
    
    # Indices for W (C): k*n + i
    # C0,2g = 2g*2 + 0 = 4g
    # C1,2g = 2g*2 + 1 = 4g+1
    # C0,2g+1 = (2g+1)*2 + 0 = 4g+2
    # C1,2g+1 = (2g+1)*2 + 1 = 4g+3
    
    for g in range(3):
        # Base indices for this group
        v_n00 = 2*g
        v_n01 = 2*g + 1
        v_n10 = 6 + 2*g
        v_n11 = 7 + 2*g
        
        w_c00 = 4*g
        w_c10 = 4*g + 1
        w_c01 = 4*g + 2
        w_c11 = 4*g + 3
        
        # P1
        r = 7*g + 0
        U[0][r] += 1
        U[3][r] += 1
        V[v_n00][r] += 1
        V[v_n11][r] += 1
        W[w_c00][r] += 1
        W[w_c10][r] += 1
        W[w_c11][r] += 1
        
        # P2
        r = 7*g + 1
        U[2][r] += 1
        U[3][r] += 1
        V[v_n00][r] += 1
        W[w_c10][r] += 1
        W[w_c11][r] -= 1
        
        # P3
        r = 7*g + 2
        U[0][r] += 1
        V[v_n10][r] += 1
        V[v_n11][r] -= 1
        W[w_c01][r] += 1
        W[w_c11][r] += 1
        
        # P4
        r = 7*g + 3
        U[3][r] += 1
        V[v_n01][r] += 1
        V[v_n00][r] -= 1
        W[w_c00][r] += 1
        W[w_c10][r] += 1
        
        # P5
        r = 7*g + 4
        U[0][r] += 1
        U[1][r] += 1
        V[v_n11][r] += 1
        W[w_c00][r] -= 1
        W[w_c01][r] += 1
        
        # P6
        r = 7*g + 5
        U[2][r] += 1
        U[0][r] -= 1
        V[v_n00][r] += 1
        V[v_n10][r] += 1
        W[w_c11][r] += 1
        
        # P7
        r = 7*g + 6
        U[1][r] += 1
        U[3][r] -= 1
        V[v_n01][r] += 1
        V[v_n11][r] += 1
        W[w_c00][r] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
