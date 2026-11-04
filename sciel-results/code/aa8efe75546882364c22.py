"""Matrix multiplication 4,5,5 - tensor decomposition."""
import time

DIMENSIONS = [4, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 140

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5x5 matrix multiplication."""
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    # Schoolbook decomposition: rank = n*m*p = 100
    # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
