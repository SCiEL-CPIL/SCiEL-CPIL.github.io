DIMENSIONS = [2, 3, 10]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition for matrix multiplication M_{n,m,p}
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[i,j,k] = 1 for all i,j,k
    # Decomposition: T = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For schoolbook: one term per (i,j,k) triple
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
