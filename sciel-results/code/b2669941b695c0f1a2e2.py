"""Program for 4x4x7 matrix multiplication tensor decomposition using block-structured schoolbook."""

def construct():
    """
    Returns an exact tensor decomposition for multiplying a 4x4 matrix by a 4x7 matrix.
    Uses a block-structured approach: split the 7 columns into 4+3, applying
    schoolbook decomposition to each block. Total rank = 4*4*4 + 4*4*3 = 64 + 48 = 112.
    
    The output factor follows the official trace convention W[k*n+i, r].
    """
    n, m, p = 4, 4, 7
    rank = n * m * p  # 112
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
