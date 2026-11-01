"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 10
    num_groups = p // 2  # 5 groups of 2 columns
    rank = num_groups * 7  # 35
    
    U = [[0] * rank for _ in range(n * m)]  # 4 x 35
    V = [[0] * rank for _ in range(m * p)]  # 20 x 35
    W = [[0] * rank for _ in range(p * n)]  # 20 x 35
    
    for g in range(num_groups):
        base = g * 7
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # M1: (a+d)(e+h)
        c = base + 0
        U[0][c] = 1
        U[3][c] = 1
        V[k0][c] = 1
        V[10 + k1][c] = 1
        W[4*g][c] = 1
        W[4*g+3][c] = 1
        
        # M2: (c+d)e
        c = base + 1
        U[2][c] = 1
        U[3][c] = 1
        V[k0][c] = 1
        W[4*g+1][c] = 1
        W[4*g+3][c] = -1
        
        # M3: a(f-h)
        c = base + 2
        U[0][c] = 1
        V[k1][c] = 1
        V[10 + k1][c] = -1
        W[4*g+2][c] = 1
        W[4*g+3][c] = 1
        
        # M4: d(g-e)
        c = base + 3
        U[3][c] = 1
        V[10 + k0][c] = 1
        V[k0][c] = -1
        W[4*g][c] = 1
        W[4*g+1][c] = 1
        
        # M5: (a+b)h
        c = base + 4
        U[0][c] = 1
        U[1][c] = 1
        V[10 + k1][c] = 1
        W[4*g][c] = -1
        W[4*g+2][c] = 1
        
        # M6: (c-a)(e+f)
        c = base + 5
        U[2][c] = 1
        U[0][c] = -1
        V[k0][c] = 1
        V[k1][c] = 1
        W[4*g+3][c] = 1
        
        # M7: (b-d)(g+h)
        c = base + 6
        U[1][c] = 1
        U[3][c] = -1
        V[10 + k0][c] = 1
        V[10 + k1][c] = 1
        W[4*g][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
