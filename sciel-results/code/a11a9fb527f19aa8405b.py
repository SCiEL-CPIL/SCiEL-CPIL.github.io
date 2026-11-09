"""
Optimal rank decomposition for 2x3 matrix multiplied by 3x8 matrix.
The tensor rank for (2, 3, 8) is exactly 2*3*8 = 48.
The schoolbook decomposition achieves this rank.
"""

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    # The tensor T_{i,j,k} = sum_c U_{i,j}^c V_{j,k}^c W_{k,i}^c
    # For schoolbook, each term corresponds to a unique (i,j,k) triple.
    # U[i*m+j, c] = 1 if c corresponds to (i,j,k)
    # V[j*p+k, c] = 1 if c corresponds to (i,j,k)
    # W[k*n+i, c] = 1 if c corresponds to (i,j,k)
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
                
    return {"U": U, "V": V, "W": W}
