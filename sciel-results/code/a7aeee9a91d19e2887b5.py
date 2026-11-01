"""Improved program using block-diagonal decomposition for matrix multiplication tensor."""
DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank is n * m, using the fact that M_{1,m,p} has rank m
    # Total rank = n * m
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for j in range(m):
            r = i * m + j
            # U index: i*m + j
            U[i * m + j][r] = 1
            
            # V index: j'*p + k. Non-zero if j' == j
            for k in range(p):
                V[j * p + k][r] = 1
            
            # W index: k'*n + i'. Non-zero if i' == i
            for k in range(p):
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
