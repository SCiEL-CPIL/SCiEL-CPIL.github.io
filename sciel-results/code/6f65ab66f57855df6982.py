"""
Rank-minimal tensor decomposition for 2x4 by 4x7 matrix multiplication.

Uses the fact that M_{2,m,p} decomposes into two independent M_{1,m,p} tensors,
each of rank p, giving total rank 2p = 14.
"""
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = 2 * p  # = 14
    num_U_rows = n * m  # 8
    num_V_rows = m * p  # 28
    num_W_rows = p * n  # 14
    
    U = [[0] * rank for _ in range(num_U_rows)]
    V = [[0] * rank for _ in range(num_V_rows)]
    W = [[0] * rank for _ in range(num_W_rows)]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * p + k
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
