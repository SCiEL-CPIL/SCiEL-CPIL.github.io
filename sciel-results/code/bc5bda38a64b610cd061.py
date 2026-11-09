import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    n, m, p = 2, 4, 5

    # Strassen decomposition for 2x2x2
    strassen_U = [
        [1,0,0,1], [0,0,1,1], [1,0,0,0], [0,0,0,1],
        [1,1,0,0], [-1,0,1,0], [0,1,0,-1]
    ]
    strassen_V = [
        [1,0,0,1], [1,0,0,0], [0,1,0,-1], [-1,0,1,0],
        [0,0,0,1], [1,1,0,0], [0,0,1,1]
    ]
    strassen_W = [
        [1,0,0,1], [0,1,0,-1], [0,0,1,1], [1,1,0,0],
        [-1,0,1,0], [0,0,0,1], [1,0,0,0]
    ]

    # Rank: 4 Strassen blocks (28) + 1 separable term for the last column (1) = 29
    rank = 4 * 7 + 1
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    # 4 Strassen blocks for the 2x4x4 part (columns 0..3 of B)
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            for r in range(7):
                # U part: 2x2 block at rows 0-1, cols j0..j0+1
                for li in range(2):
                    for lj in range(2):
                        gi = li
                        gj = j0 + lj
                        U[gi * m + gj][col] = strassen_U[r][li * 2 + lj]
                # V part: 2x2 block at rows j0..j0+1, cols k0..k0+1
                for lj in range(2):
                    for lk in range(2):
                        gj = j0 + lj
                        gk = k0 + lk
                        V[gj * p + gk][col] = strassen_V[r][lj * 2 + lk]
                # W part: 2x2 block at rows k0..k0+1, cols 0-1
                for lk in range(2):
                    for li in range(2):
                        gk = k0 + lk
                        gi = li
                        W[gk * n + gi][col] = strassen_W[r][lk * 2 + li]
                col += 1

    # Single separable term for the 2x4x1 part (last column of B, k=4)
    # T[i,j,4] = 1 for all i,j; T[i,j,k] = 0 for k != 4
    # U[i,j] = 1, V[j,4] = 1, W[4,i] = 1
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
    for j in range(m):
        V[j * p + 4][col] = 1
    for i in range(n):
        W[4 * n + i][col] = 1

    return {"U": U, "V": V, "W": W}
