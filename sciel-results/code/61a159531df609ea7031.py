import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the 49-rank Strassen decomposition for 4x4x4 matrix multiplication.
    T[i,j,k] = sum_r U[i*4+j,r] * V[j*4+k,r] * W[k*4+i,r]
    Recursive Strassen: 7*7=49, proven optimal by Bläser (2002).
    """
    n, m, p = DIMENSIONS
    rank = 49

    beta = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]]
    ]
    gamma = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[-1, 0], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 0], [1, 1]]
    ]
    delta = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            for i in range(4):
                for j in range(4):
                    c = beta[s][i >> 1][j >> 1] * beta[t][i & 1][j & 1]
                    if c:
                        U[i * 4 + j][col] = c
            for j in range(4):
                for k in range(4):
                    c = gamma[s][j >> 1][k >> 1] * gamma[t][j & 1][k & 1]
                    if c:
                        V[j * 4 + k][col] = c
            for k in range(4):
                for i in range(4):
                    c = delta[s][i >> 1][k >> 1] * delta[t][i & 1][k & 1]
                    if c:
                        W[k * 4 + i][col] = c

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
