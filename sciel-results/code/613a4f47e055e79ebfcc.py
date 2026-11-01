"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-f)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U indices: i*m+j (0:a, 1:b, 2:c, 3:d)
    # V indices: j*p+k
    # W indices: k*n+i
    
    for b in range(8):
        base = 7 * b
        # k indices for this block: 2b, 2b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # V rows for e, f, g, h
        # e = B[0, k0], f = B[0, k1], g = B[1, k0], h = B[1, k1]
        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1
        
        # W rows for C00, C01, C10, C11
        # C00 = C[k0, 0], C01 = C[k0, 1], C10 = C[k1, 0], C11 = C[k1, 1]
        w_00 = k0 * n + 0
        w_01 = k0 * n + 1
        w_10 = k1 * n + 0
        w_11 = k1 * n + 1
        
        # Term 1: M1 = (a+d)(e+h)
        r = base
        U[0][r] = 1; U[3][r] = 1
        V[v_e][r] = 1; V[v_h][r] = 1
        W[w_00][r] = 1; W[w_11][r] = 1
        
        # Term 2: M2 = (c+d)e
        r = base + 1
        U[2][r] = 1; U[3][r] = 1
        V[v_e][r] = 1
        W[w_10][r] = 1; W[w_11][r] = -1
        
        # Term 3: M3 = a(f-h)
        r = base + 2
        U[0][r] = 1
        V[v_f][r] = 1; V[v_h][r] = -1
        W[w_01][r] = 1; W[w_11][r] = 1
        
        # Term 4: M4 = d(g-f)
        r = base + 3
        U[3][r] = 1
        V[v_g][r] = 1; V[v_f][r] = -1
        W[w_00][r] = 1; W[w_10][r] = 1
        
        # Term 5: M5 = (a+b)h
        r = base + 4
        U[0][r] = 1; U[1][r] = 1
        V[v_h][r] = 1
        W[w_00][r] = -1; W[w_01][r] = 1
        
        # Term 6: M6 = (c-a)(e+f)
        r = base + 5
        U[2][r] = 1; U[0][r] = -1
        V[v_e][r] = 1; V[v_f][r] = 1
        W[w_11][r] = 1
        
        # Term 7: M7 = (b-d)(g+h)
        r = base + 6
        U[1][r] = 1; U[3][r] = -1
        V[v_g][r] = 1; V[v_h][r] = 1
        W[w_00][r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
