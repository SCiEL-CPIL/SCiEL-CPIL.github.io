def _strassen_4x4x4():
    """Build rank-64 schoolbook decomposition for 4x4x4.
    Returns (U, V, W) of shape (16, 64) each.
    """
    rank = 64
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                U[i*4+j][col] = 1
                V[j*4+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
    return U, V, W
