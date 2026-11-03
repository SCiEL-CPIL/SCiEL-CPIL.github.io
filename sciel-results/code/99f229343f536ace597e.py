# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Row-vector parallel decomposition
    # Each row i of A (1x4) multiplies B (4x9) to produce row i of C (1x9)
    # This sub-problem is decomposed into m*p terms per row (naive rank)
    # Total rank = n * m * p
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            for j in range(m):
                # Determine term index
                # We map (i, k, j) to a unique index
                # i: 0..3, k: 0..8, j: 0..3
                # index = (i * p + k) * m + j
                idx = (i * p + k) * m + j
                U[i * m + j][idx] = 1
                V[j * p + k][idx] = 1
                W[k * n + i][idx] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
