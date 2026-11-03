import time

def construct():
    n, m, p = 4, 4, 6
    rank = 24
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 24
    V = [[0] * rank for _ in range(m * p)]  # 24 x 24
    W = [[0] * rank for _ in range(p * n)]  # 24 x 24
    
    for j in range(p):  # j = 0..5 (column of C / B)
        for k in range(m):  # k = 0..3 (shared index)
            r = j * m + k  # rank index
            for i in range(n):  # i = 0..3 (row of C / A)
                U[i * m + k][r] = 1  # A[i,k]
                W[j * n + i][r] = 1  # C[i,j]
            V[k * p + j][r] = 1  # B[k,j]
    
    return {"U": U, "V": V, "W": W}
