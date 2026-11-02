DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen for 2x2 by 2x2 blocks
    # Rank 7 for each block
    # 4 blocks -> Rank 28
    
    # Strassen coefficients for 2x2 by 2x2
    # U, V, W are 4x7 matrices (flattened 2x2)
    # Indices: i*2+j, j*2+k, k*2+i
    
    # M1
    U1 = [1, 0, 0, 1]
    V1 = [1, 0, 0, 1]
    W1 = [1, 0, 0, 1]
    
    # M2
    U2 = [0, 0, 1, 1]
    V2 = [1, 0, 0, 0]
    W2 = [0, 1, 0, 1]
    
    # M3
    U3 = [1, 0, 0, 0]
    V3 = [0, 1, 0, -1]
    W3 = [0, 0, 1, 1]
    
    # M4
    U4 = [0, 0, 0, 1]
    V4 = [0, 0, 1, -1]
    W4 = [1, 1, 0, 0]
    
    # M5
    U5 = [1, 1, 0, 0]
    V5 = [0, 0, 0, 1]
    W5 = [-1, 0, 1, 0]
    
    # M6
    U6 = [-1, 0, 1, 0]
    V6 = [1, 1, 0, 0]
    W6 = [0, 0, 0, 1]
    
    # M7
    U7 = [0, 1, 0, -1]
    V7 = [0, 0, 1, 1]
    W7 = [1, 0, 0, 1]
    
    strassen_U = [U1, U2, U3, U4, U5, U6, U7]
    strassen_V = [V1, V2, V3, V4, V5, V6, V7]
    strassen_W = [W1, W2, W3, W4, W5, W6, W7]
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: A_L (rows 0-1, cols 0-1) x B_LL (rows 0-1, cols 0-1)
    # Global i: 0-1, j: 0-1, k: 0-1
    for a in range(7):
        for local_i in range(2):
            for local_j in range(2):
                U[local_i * m + local_j][a] = strassen_U[a][local_i * 2 + local_j]
        for local_j in range(2):
            for local_k in range(2):
                V[local_j * p + local_k][a] = strassen_V[a][local_j * 2 + local_k]
        for local_k in range(2):
            for local_i in range(2):
                W[local_k * n + local_i][a] = strassen_W[a][local_k * 2 + local_i]
                
    # Block 2: A_R (rows 0-1, cols 2-3) x B_RL (rows 2-3, cols 0-1)
    # Global i: 0-1, j: 2-3, k: 0-1
    offset = 7
    for a in range(7):
        for local_i in range(2):
            for local_j in range(2):
                U[local_i * m + (2 + local_j)][offset + a] = strassen_U[a][local_i * 2 + local_j]
        for local_j in range(2):
            for local_k in range(2):
                V[(2 + local_j) * p + local_k][offset + a] = strassen_V[a][local_j * 2 + local_k]
        for local_k in range(2):
            for local_i in range(2):
                W[local_k * n + local_i][offset + a] = strassen_W[a][local_k * 2 + local_i]

    # Block 3: A_L (rows 0-1, cols 0-1) x B_LR (rows 0-1, cols 2-3)
    # Global i: 0-1, j: 0-1, k: 2-3
    offset = 14
    for a in range(7):
        for local_i in range(2):
            for local_j in range(2):
                U[local_i * m + local_j][offset + a] = strassen_U[a][local_i * 2 + local_j]
        for local_j in range(2):
            for local_k in range(2):
                V[local_j * p + (2 + local_k)][offset + a] = strassen_V[a][local_j * 2 + local_k]
        for local_k in range(2):
            for local_i in range(2):
                W[(2 + local_k) * n + local_i][offset + a] = strassen_W[a][local_k * 2 + local_i]

    # Block 4: A_R (rows 0-1, cols 2-3) x B_RR (rows 2-3, cols 2-3)
    # Global i: 0-1, j: 2-3, k: 2-3
    offset = 21
    for a in range(7):
        for local_i in range(2):
            for local_j in range(2):
                U[local_i * m + (2 + local_j)][offset + a] = strassen_U[a][local_i * 2 + local_j]
        for local_j in range(2):
            for local_k in range(2):
                V[(2 + local_j) * p + (2 + local_k)][offset + a] = strassen_V[a][local_j * 2 + local_k]
        for local_k in range(2):
            for local_i in range(2):
                W[(2 + local_k) * n + local_i][offset + a] = strassen_W[a][local_k * 2 + local_i]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
