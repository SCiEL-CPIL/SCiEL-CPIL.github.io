DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = m  # 5
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for i in range(n):
            for j1 in range(m):
                U[i * m + j1][c] = 1 if j1 == c else 0
        
        for j2 in range(m):
            for k in range(p):
                V[j2 * p + k][c] = 1 if j2 == c else 0
        
        for k in range(p):
            for i in range(n):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
