DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m  # 20
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U is identity matrix of size n*m
    for idx in range(n * m):
        U[idx][idx] = 1
    
    # V[j*p+k, i*m+j] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                V[j * p + k][i * m + j] = 1
    
    # W[k*n+i, i*m+j] = 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                W[k * n + i][i * m + j] = 1
    
    return {"U": U, "V": V, "W": W}
