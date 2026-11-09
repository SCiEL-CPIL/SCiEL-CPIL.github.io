"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Split m into 2 and 1
    m1, m2 = 2, 1
    
    # Rank 1 part for j=2 (m2)
    # U[i*m2+0, r] = 1
    # V[0*p+k, r] = 1
    # W[k*n+i, r] = 1
    
    # Rank 28 part for j=0,1 (m1)
    # For each j in 0,1, rank 1? No, rank 4 per j?
    # 2x2x7 is 28.
    
    # Let's just use schoolbook for the 2x2x7 part to be safe?
    # Rank 28.
    # Total 29.
    
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: j=2
    for i in range(n):
        U[i * m + 2][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    # Part 2: j=0, 1
    # Schoolbook for 2x2x7
    for i in range(n):
        for j in range(m1):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
