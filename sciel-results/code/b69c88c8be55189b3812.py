"""Matrix multiplication 2x4 by 4x6 tensor decomposition."""

def construct():
    n, m, p = 2, 4, 6
    # Schoolbook rank = 48
    U = []
    V = []
    W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U has shape (n*m, rank), index i*m+j
                U.append([0.0] * 48)
                U[i*m+j][len(U)-1] = 1.0  # placeholder, we'll fix indexing
    # Actually let's build it properly
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
    return {"U": U, "V": V, "W": W}
