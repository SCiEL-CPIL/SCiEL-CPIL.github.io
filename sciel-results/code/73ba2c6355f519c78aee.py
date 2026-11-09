def construct():
    n, m, p = 2, 5, 5
    # Block decomposition: split index 4 from 0-3
    # Parts:
    # 1. 2x4 x 4x4 -> 2x4 (rank 8)
    # 2. 2x4 x 4x1 -> 2x1 (rank 8)
    # 3. 2x1 x 1x4 -> 2x4 (rank 8)
    # 4. 2x1 x 1x1 -> 2x1 (rank 2)
    # Total rank: 26
    
    rank = 26
    U = [[0]*rank for _ in range(n*m)]  # 10 x 26
    V = [[0]*rank for _ in range(m*p)]  # 25 x 26
    W = [[0]*rank for _ in range(p*n)]  # 10 x 26
    
    col = 0
    # Part 1: 2x4 x 4x4 -> 2x4
    # A[i,j] * B[j,k] for i in 0,1; j in 0-3; k in 0-3
    for i in range(2):
        for j in range(4):
            for k in range(4):
                U[i*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i][col] = 1
                col += 1
    
    # Part 2: 2x4 x 4x1 -> 2x1
    # A[i,j] * B[j,4] for i in 0,1; j in 0-3
    for i in range(2):
        for j in range(4):
            U[i*m + j][col] = 1
            V[j*p + 4][col] = 1
            W[4*n + i][col] = 1
            col += 1
    
    # Part 3: 2x1 x 1x4 -> 2x4
    # A[i,4] * B[4,k] for i in 0,1; k in 0-3
    for i in range(2):
        for k in range(4):
            U[i*m + 4][col] = 1
            V[4*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
    
    # Part 4: 2x1 x 1x1 -> 2x1
    # A[i,4] * B[4,4] for i in 0,1
    for i in range(2):
        U[i*m + 4][col] = 1
        V[4*p + 4][col] = 1
        W[4*n + i][col] = 1
        col += 1
    
    return {"U": U, "V": V, "W": W}
