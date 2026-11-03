DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Combined terms for 4 (j,k) pairs
    combined_jk = [(0, 0), (1, 0), (2, 0), (3, 0)]
    col = 0
    for j, k in combined_jk:
        U[j][col] = 1
        U[m + j][col] = 1
        V[j * p + k][col] = 1
        W[k * n][col] = 1
        W[k * n + 1][col] = 1
        col += 1
    
    # Standard terms for remaining entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (j, k) in combined_jk:
                    continue
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
