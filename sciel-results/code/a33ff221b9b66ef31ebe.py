"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Block 1: A1 (cols 0,1) * B1 (rows 0,1)
    for i in range(n):
        for k in range(p):
            # a_{i,0} * b_{0,k}
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
            # a_{i,1} * b_{1,k}
            U[i * m + 1][col] = 1
            V[1 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    # Block 2: A2 (cols 2,3) * B2 (rows 2,3)
    for i in range(n):
        for k in range(p):
            # a_{i,2} * b_{2,k}
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
            # a_{i,3} * b_{3,k}
            U[i * m + 3][col] = 1
            V[3 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
