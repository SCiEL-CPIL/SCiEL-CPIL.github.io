def construct():
    n, m, p = 3, 3, 5
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: 3x3x3 using Schoolbook (Rank 27)
    # Indices 0-26
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3): # First 3 columns
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    # Part 2: 3x3x2 using Schoolbook (Rank 6)
    # Indices 27-32
    for i in range(n):
        for j in range(m):
            for k in range(3, 5): # Last 2 columns
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    return {"U": U, "V": V, "W": W}
