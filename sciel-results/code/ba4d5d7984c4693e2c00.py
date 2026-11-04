"""Rewritten program with rank-3 decomposition."""
DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is m
    rank = m
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each j in 0..m-1
    for j in range(m):
        # U column j
        # U[i*m + j', j] = 1 if j' == j else 0
        for i in range(n):
            U[i * m + j][j] = 1
            
        # V column j
        # V[j'*p + k, j] = 1 if j' == j else 0
        for k in range(p):
            V[j * p + k][j] = 1
            
        # W column j
        # W[k*n + i, j] = 1 for all i, k
        for k in range(p):
            for i in range(n):
                W[k * n + i][j] = 1
                
    return {"U": U, "V": V, "W": W}
