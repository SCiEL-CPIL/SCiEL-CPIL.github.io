"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 using Strassen's algorithm for 5 blocks of 2x2
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Output reconstruction
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M7
    # C11 = M1 - M2 + M3 + M6
    
    # U indices: i*m + l (0..3)
    # V indices: l*p + k (0..19)
    # W indices: k*n + i (0..19)
    
    for j in range(5):
        base_col = 7 * j
        # Block j corresponds to B columns 2j, 2j+1
        # B entries:
        # e = B[0, 2j] -> V[0*p + 2j]
        # f = B[0, 2j+1] -> V[0*p + 2j+1]
        # g = B[1, 2j] -> V[1*p + 2j]
        # h = B[1, 2j+1] -> V[1*p + 2j+1]
        
        # W entries for this block:
        # C00 = C[0, 2j] -> W[2j*n + 0]
        # C01 = C[0, 2j+1] -> W[(2j+1)*n + 0]
        # C10 = C[1, 2j] -> W[2j*n + 1]
        # C11 = C[1, 2j+1] -> W[(2j+1)*n + 1]
        
        w_00 = 2j * n + 0
        w_01 = (2j + 1) * n + 0
        w_10 = 2j * n + 1
        w_11 = (2j + 1) * n + 1
        
        v_e = 0 * p + 2j
        v_f = 0 * p + 2j + 1
        v_g = 1 * p + 2j
        v_h = 1 * p + 2j + 1
        
        # M1
        col = base_col
        U[0][col] = 1 # a
        U[3][col] = 1 # d
        V[v_e][col] = 1
        V[v_h][col] = 1
        W[w_00][col] = 1
        W[w_11][col] = 1
        
        # M2
        col = base_col + 1
        U[2][col] = 1 # c
        U[3][col] = 1 # d
        V[v_e][col] = 1
        W[w_10][col] = 1
        W[w_11][col] = -1
        
        # M3
        col = base_col + 2
        U[0][col] = 1 # a
        V[v_f][col] = 1
        V[v_h][col] = -1
        W[w_01][col] = 1
        W[w_11][col] = 1
        
        # M4
        col = base_col + 3
        U[3][col] = 1 # d
        V[v_g][col] = 1
        V[v_e][col] = -1
        W[w_00][col] = -1
        W[w_10][col] = 1
        
        # M5
        col = base_col + 4
        U[0][col] = 1 # a
        U[1][col] = 1 # b
        V[v_h][col] = 1
        W[w_00][col] = -1
        W[w_01][col] = 1
        
        # M6
        col = base_col + 5
        U[2][col] = 1 # c
        U[0][col] = -1 # -a
        V[v_e][col] = 1
        V[v_f][col] = 1
        W[w_11][col] = 1
        
        # M7
        col = base_col + 6
        U[1][col] = 1 # b
        U[3][col] = -1 # -d
        V[v_g][col] = 1
        V[v_h][col] = 1
        W[w_00][col] = -1
        W[w_10][col] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
