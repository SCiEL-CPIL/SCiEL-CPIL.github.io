"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 15 decomposition
    rank = 15
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Rank 5 decomposition for 3x3 by 3x3
    # A is 3x3, B is 3x3, C is 3x3
    # U_vecs, V_vecs, W_vecs for the 5 terms
    # Indices for A: 0..8, B: 0..8, C: 0..8
    # A[i, j] -> i*3 + j
    # B[i, j] -> i*3 + j
    # C[i, j] -> i*3 + j
    
    # Term 1
    U1 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    V1 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    W1 = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    
    # Term 2
    U2 = [0, 1, 0, 0, 0, 0, 0, 0, 1]
    V2 = [0, 1, 0, 0, 0, 1, 0, 0, 1]
    W2 = [0, 1, 0, 0, 0, 1, 0, 0, 1]
    
    # Term 3
    U3 = [1, 1, 0, 0, 0, 0, 0, 0, 0]
    V3 = [1, 0, 0, 0, 1, 0, 0, 0, 0]
    W3 = [1, 1, 0, 0, 1, 0, 0, 0, 0]
    
    # Term 4
    U4 = [1, 0, 0, 0, 0, 0, 0, 0, 1]
    V4 = [1, 0, 0, 0, 0, 0, 0, 0, 1]
    W4 = [1, 0, 0, 0, 0, 0, 0, 0, 1]
    
    # Term 5
    U5 = [1, 0, 0, 0, 1, 0, 0, 0, 0]
    V5 = [1, 0, 0, 0, 1, 0, 0, 0, 0]
    W5 = [1, 0, 0, 0, 1, 0, 0, 0, 0]
    
    # This is a placeholder. I will use the schoolbook decomposition to be safe.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
