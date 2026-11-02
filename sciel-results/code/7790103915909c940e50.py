"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2 multiplication
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # U indices: 0:a, 1:b, 2:c, 3:d
    # V indices for block l:
    # l=0: 0:e, 1:f, 4:g, 5:h
    # l=1: 2:e, 3:f, 6:g, 7:h
    # W indices for block l (cols 2l, 2l+1):
    # l=0: 0:C00, 1:C10, 2:C01, 3:C11
    # l=1: 4:C02, 5:C12, 6:C03, 7:C13
    
    for l in range(2):
        base_col = l * 7
        # V rows for e, f, g, h in this block
        v_e = 0 * p + (2 * l)      # j=0, k=2l
        v_f = 0 * p + (2 * l + 1)  # j=0, k=2l+1
        v_g = 1 * p + (2 * l)      # j=1, k=2l
        v_h = 1 * p + (2 * l + 1)  # j=1, k=2l+1
        
        # W rows for C00, C10, C01, C11 in this block
        # C[i, 2l+r] -> W[2(2l+r) + i]
        w_00 = 2 * (2 * l + 0) + 0
        w_10 = 2 * (2 * l + 0) + 1
        w_01 = 2 * (2 * l + 1) + 0
        w_11 = 2 * (2 * l + 1) + 1
        
        # M1 = (a+d)(e+h)
        col = base_col + 0
        U[0][col] = 1; U[3][col] = 1
        V[v_e][col] = 1; V[v_h][col] = 1
        W[w_00][col] = 1; W[w_11][col] = 1
        
        # M2 = (c+d)e
        col = base_col + 1
        U[2][col] = 1; U[3][col] = 1
        V[v_e][col] = 1
        W[w_10][col] = 1; W[w_11][col] = -1
        
        # M3 = a(f-h)
        col = base_col + 2
        U[0][col] = 1
        V[v_f][col] = 1; V[v_h][col] = -1
        W[w_01][col] = 1; W[w_11][col] = 1
        
        # M4 = d(g-e)
        col = base_col + 3
        U[3][col] = 1
        V[v_g][col] = 1; V[v_e][col] = -1
        W[w_00][col] = -1; W[w_10][col] = 1
        
        # M5 = (a+b)h
        col = base_col + 4
        U[0][col] = 1; U[1][col] = 1
        V[v_h][col] = 1
        W[w_00][col] = 1; W[w_01][col] = 1
        
        # M6 = (c-a)(e+f)
        col = base_col + 5
        U[2][col] = 1; U[0][col] = -1
        V[v_e][col] = 1; V[v_f][col] = 1
        W[w_11][col] = 1
        
        # M7 = (b-d)(g+h)
        col = base_col + 6
        U[1][col] = 1; U[3][col] = -1
        V[v_g][col] = 1; V[v_h][col] = 1
        W[w_00][col] = -1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
