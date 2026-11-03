"""Matrix multiplication 3x4x4: find rank-minimal tensor decomposition."""
import time

DIMENSIONS = [3, 4, 4]
n, m, p = DIMENSIONS

def construct():
    """
    Construct a tensor decomposition for 3x4 * 4x4 matmul.
    Returns a dictionary with keys 'U', 'V', 'W' containing the factor matrices.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    The schoolbook decomposition is used, which has rank n*m*p = 48.
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j][column] = 1
                # V[j*p + k][column] = 1
                # W[k*n + i][column] = 1
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
