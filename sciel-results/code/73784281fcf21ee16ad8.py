"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition using 5 blocks of Strassen (2x2x2)
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 5 blocks of 2 columns
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Strassen terms for block b
        # Term 1: (a+d)(e+h)
        # U: a(0), d(3)
        # V: e(0, k0), h(1, k1)
        # W: C00(k0, 0) +, C11(k1, 1) +
        U[0][col] = 1; U[3][col] = 1
        V[0 * 10 + k0][col] = 1; V[1 * 10 + k1][col] = 1
        W[k0 * 2 + 0][col] = 1; W[k1 * 2 + 1][col] = 1
        col += 1
        
        # Term 2: (c+d)e
        # U: c(2), d(3)
        # V: e(0, k0)
        # W: C10(k0, 1) +, C11(k1, 1) -
        U[2][col] = 1; U[3][col] = 1
        V[0 * 10 + k0][col] = 1
        W[k0 * 2 + 1][col] = 1; W[k1 * 2 + 1][col] = -1
        col += 1
        
        # Term 3: a(f-h)
        # U: a(0)
        # V: f(0, k1), h(1, k1)
        # W: C01(k1, 0) +, C11(k1, 1) +
        U[0][col] = 1
        V[0 * 10 + k1][col] = 1; V[1 * 10 + k1][col] = -1
        W[k1 * 2 + 0][col] = 1; W[k1 * 2 + 1][col] = 1
        col += 1
        
        # Term 4: d(g-e)
        # U: d(3)
        # V: g(1, k0), e(0, k0)
        # W: C00(k0, 0) +, C10(k0, 1) +
        U[3][col] = 1
        V[1 * 10 + k0][col] = 1; V[0 * 10 + k0][col] = -1
        W[k0 * 2 + 0][col] = 1; W[k0 * 2 + 1][col] = 1
        col += 1
        
        # Term 5: (a+b)h
        # U: a(0), b(1)
        # V: h(1, k1)
        # W: C00(k0, 0) -, C01(k1, 0) +
        U[0][col] = 1; U[1][col] = 1
        V[1 * 10 + k1][col] = 1
        W[k0 * 2 + 0][col] = -1; W[k1 * 2 + 0][col] = 1
        col += 1
        
        # Term 6: (c-a)(e+f)
        # U: c(2), a(0)
        # V: e(0, k0), f(0, k1)
        # W: C11(k1, 1) +
        U[2][col] = 1; U[0][col] = -1
        V[0 * 10 + k0][col] = 1; V[0 * 10 + k1][col] = 1
        W[k1 * 2 + 1][col] = 1
        col += 1
        
        # Term 7: (b-d)(g+h)
        # U: b(1), d(3)
        # V: g(1, k0), h(1, k1)
        # W: C00(k0, 0) +
        U[1][col] = 1; U[3][col] = -1
        V[1 * 10 + k0][col] = 1; V[1 * 10 + k1][col] = 1
        W[k0 * 2 + 0][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
