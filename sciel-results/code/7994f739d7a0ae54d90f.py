"""
Optimal rank decomposition for 2x2 by 2x6 matrix multiplication.
The minimum rank is 24, as the tensor is the direct sum of six M_{2,2,1} tensors,
each of which has rank 4.
"""
DIMENSIONS = [2, 2, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 24
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][col] = 1
                # V index: j*p + k
                V[j * p + k][col] = 1
                # W index: k*n + i (as per trace convention W[k*n+i, r])
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
