DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Rank = m * p = 30
    # Structure: For each j in m and k in p, create a factor.
    # V picks out B[j,k]
    # U picks out A[i,j] (for all i)
    # W distributes to C[i,k] (for all i)
    rank = m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for j in range(m):
        for k in range(p):
            # V[j*p + k, col] = 1
            V[j * p + k][col] = 1
            # U[i*m + j, col] = 1 for all i
            for i in range(n):
                U[i * m + j][col] = 1
                # W[k*n + i, col] = 1
                W[k * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
