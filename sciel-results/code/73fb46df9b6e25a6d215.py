"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 8 x 36
    V = [[0] * rank for _ in range(m * p)]  # 20 x 36
    W = [[0] * rank for _ in range(p * n)]  # 10 x 36
    col = 0
    
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]
    
    def add_strassen_block(i1, j1, k1):
        nonlocal col
        for u_pat, v_pat, w_pat in strassen_terms:
            for (di, dj), coeff in u_pat.items():
                U[(i1+di)*m + (j1+dj)][col] += coeff
            for (dj, dk), coeff in v_pat.items():
                V[(j1+dj)*p + (k1+dk)][col] += coeff
            for (di, dk), coeff in w_pat.items():
                W[(k1+dk)*n + (i1+di)][col] += coeff
            col += 1
    
    def add_schoolbook_2x2x1(i1, j1, k1):
        nonlocal col
        for di in range(2):
            for dj in range(2):
                i = i1 + di
                j = j1 + dj
                k = k1
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 2, 2)
    add_schoolbook_2x2x1(0, 0, 4)
    add_schoolbook_2x2x1(0, 2, 4)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
