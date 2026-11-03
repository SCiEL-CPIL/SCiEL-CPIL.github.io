DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    # Schoolbook per block: 4x4x4 has rank 16, 4x4x1 has rank 4
    # Total: 16 + 16 + 4 = 36
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # (16, 36)
    V = [[0] * rank for _ in range(m * p)]  # (36, 36)
    W = [[0] * rank for _ in range(p * n)]  # (36, 36)
    
    col = 0
    # Block 1: A(4x4) x B[:,0:4](4x4) -> C[:,0:4](4x4)
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # Block 2: A(4x4) x B[:,4:8](4x4) -> C[:,4:8](4x4)
    for i in range(n):
        for j in range(m):
            for k in range(4, 8):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    # Block 3: A(4x4) x B[:,8:9](4x1) -> C[:,8:9](4x1)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1
            W[8 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
