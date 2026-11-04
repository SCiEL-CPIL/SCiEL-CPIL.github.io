"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Strategy: Split m=3 into m1=2 and m2=1.
    # Part 1: 2x2x5 schoolbook (rank 20)
    # Part 2: 2x1x5 rank 1
    # Total rank 21
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 2: j=2 (the last column of A, last row of B)
    # This contributes to T[i, 2, k] = 1
    # U[i*3 + 2, col] = 1
    # V[2*5 + k, col] = 1
    # W[k*2 + i, col] = 1
    for i in range(n):
        U[i * m + 2][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    # Part 1: j in {0, 1}
    # Schoolbook for 2x2x5
    for i in range(n):
        for j in range(2):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
