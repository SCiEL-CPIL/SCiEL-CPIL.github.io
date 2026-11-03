"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 40 decomposition
    # Part 1: 2x2x10 (j=0,1) -> Rank 20
    # Part 2: 2x1x10 (j=2)   -> Rank 20
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: j in {0, 1}
    # C[i,k] += A[i,0] * B[0,k] + A[i,1] * B[1,k]
    for k in range(p):
        # Term for A[i,0] * B[0,k]
        r = 2 * k
        for i in range(n):
            U[i * m + 0][r] = 1
            W[k * n + i][r] = 1
        V[0 * p + k][r] = 1
        
        # Term for A[i,1] * B[1,k]
        r = 2 * k + 1
        for i in range(n):
            U[i * m + 1][r] = 1
            W[k * n + i][r] = 1
        V[1 * p + k][r] = 1

    # Part 2: j = 2
    # C[i,k] += A[i,2] * B[2,k]
    for k in range(p):
        for i in range(n):
            r = 20 + k * n + i
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
