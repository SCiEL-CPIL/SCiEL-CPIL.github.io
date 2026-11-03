"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of rank 32: 16 for 4x4 schoolbook + 16 for 5th row/col interactions
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # 4x4 schoolbook part
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i * 5 + k][col] = 1
                V[k * 5 + j][col] = 1
                W[i * 5 + j][col] = 1
                col += 1
    
    # 5th row/col interactions: A[i,4] * B[4,j] contributes to C[i,j]
    for i in range(4):
        for j in range(4):
            U[i * 5 + 4][col] = 1
            V[4 * 5 + j][col] = 1
            W[i * 5 + j][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
