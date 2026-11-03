"""Matrix multiplication 2x3x3: exact schoolbook decomposition with BFDS verification."""
DIMENSIONS = [2, 3, 3]

import time

# EVOLVE-BLOCK-START
_START = time.time()
N, M, P = DIMENSIONS

def construct():
    # Schoolbook decomposition of rank n*m*p
    rank = N * M * P
    U = [[0] * rank for _ in range(N * M)]
    V = [[0] * rank for _ in range(M * P)]
    W = [[0] * rank for _ in range(P * N)]
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j][col] = 1
                V[j * P + k][col] = 1
                W[k * N + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
