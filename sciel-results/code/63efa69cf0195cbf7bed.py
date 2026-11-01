"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen's algorithm for 2x2 * 2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U indices: 0: a, 1: b, 2: c, 3: d
    # V indices for block k (cols 2k, 2k+1):
    # e -> B[0, 2k] -> V[2k]
    # f -> B[0, 2k+1] -> V[2k+1]
    # g -> B[1, 2k] -> V[14+2k]
    # h -> B[1, 2k+1] -> V[14+2k+1]
    # W indices for block k (cols 2k, 2k+1):
    # C11 -> C[0, 2k] -> W[2k*2 + 0] = W[4k]
    # C12 -> C[0, 2k+1] -> W[2k*2 + 1] = W[4k+1]
    # C21 -> C[1, 2k] -> W[2k*2 + 2] = W[4k+2]
    # C22 -> C[1, 2k+1] -> W[2k*2 + 3] = W[4k+3]
    
    for k in range(7):
        col_base = k * 7
        e_idx = 2 * k
        f_idx = 2 * k + 1
        g_idx = 14 + 2 * k
        h_idx = 14 + 2 * k + 1
        
        c11_idx = 4 * k
        c12_idx = 4 * k + 1
        c21_idx = 4 * k + 2
        c22_idx = 4 * k + 3
        
        # M1: (a+d)(e+h)
        U[0][col_base] = 1
        U[3][col_base] = 1
        V[e_idx][col_base] = 1
        V[h_idx][col_base] = 1
        W[c11_idx][col_base] = 1
        W[c22_idx][col_base] = 1
        
        # M2: (c+d)e
        U[2][col_base + 1] = 1
        U[3][col_base + 1] = 1
        V[e_idx][col_base + 1] = 1
        W[c21_idx][col_base + 1] = 1
        W[c22_idx][col_base + 1] = -1
        
        # M3: a(f-h)
        U[0][col_base + 2] = 1
        V[f_idx][col_base + 2] = 1
        V[h_idx][col_base + 2] = -1
        W[c12_idx][col_base + 2] = 1
        W[c22_idx][col_base + 2] = 1
        
        # M4: d(g-e)
        U[3][col_base + 3] = 1
        V[g_idx][col_base + 3] = 1
        V[e_idx][col_base + 3] = -1
        W[c11_idx][col_base + 3] = 1
        W[c21_idx][col_base + 3] = 1
        
        # M5: (a+b)h
        U[0][col_base + 4] = 1
        U[1][col_base + 4] = 1
        V[h_idx][col_base + 4] = 1
        W[c11_idx][col_base + 4] = -1
        W[c12_idx][col_base + 4] = 1
        
        # M6: (c-a)(e+f)
        U[2][col_base + 5] = 1
        U[0][col_base + 5] = -1
        V[e_idx][col_base + 5] = 1
        V[f_idx][col_base + 5] = 1
        W[c22_idx][col_base + 5] = 1
        
        # M7: (b-d)(g+h)
        U[1][col_base + 6] = 1
        U[3][col_base + 6] = -1
        V[g_idx][col_base + 6] = 1
        V[h_idx][col_base + 6] = 1
        W[c11_idx][col_base + 6] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
