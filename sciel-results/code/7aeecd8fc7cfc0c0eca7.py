DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    # Decompose into p independent n×m × m×1 problems
    # Each has rank m (using the identity-like decomposition)
    # Total rank = p * m
    rank = p * m  # 6 * 4 = 24
    
    U = [[0] * rank for _ in range(n * m)]  # (16, 24)
    V = [[0] * rank for _ in range(m * p)]  # (24, 24)
    W = [[0] * rank for _ in range(p * n)]  # (24, 24)
    
    for k in range(p):
        base = k * m  # offset for this output column
        for i in range(n):
            for j in range(m):
                col = base + j
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
