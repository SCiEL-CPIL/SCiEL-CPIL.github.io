"""Tensor decomposition for 2x3x8 matrix multiplication using a rank-n*m construction."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        i = r // m
        j = r % m
        
        # U index: i * m + j
        U[i * m + j][r] = 1
        
        # V index: j * p + k for all k
        for k in range(p):
            V[j * p + k][r] = 1
            
        # W index: k * n + i for all k
        for k in range(p):
            W[k * n + i][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
