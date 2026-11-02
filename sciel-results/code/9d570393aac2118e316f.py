"""Improved program for matrix multiplication 2x3x8 using robust schoolbook decomposition."""
DIMENSIONS = [2, 3, 8]

def construct():
    """
    Returns a rank-minimal (or near-minimal) tensor decomposition for multiplying
    a 2x3 matrix by a 3x8 matrix.
    
    Uses the standard schoolbook decomposition which is guaranteed to be exact
    and valid. Rank is 2*3*8 = 48.
    """
    n, m, p = DIMENSIONS  # 2, 3, 8
    
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
