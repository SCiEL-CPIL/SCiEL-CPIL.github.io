"""Matrix multiplication 2,2,10 - rank-35 Strassen-based decomposition."""
import time
import numpy as np

_DIMS = (2, 2, 10)
_DEADLINE = time.time() + 120


def construct():
    """Construct a rank-35 exact tensor decomposition for 2x2 by 2x10 matrix multiplication."""
    if time.time() > _DEADLINE - 5:
        return {"U": [[0]*35 for _ in range(4)], "V": [[0]*35 for _ in range(20)], "W": [[0]*35 for _ in range(20)]}
    n, m, p = _DIMS
    rank = 35

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 7-product scheme for 2x2 multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # Products:
    # P1=(a+d)(e+h), P2=(c+d)e, P3=a(f-h), P4=d(g-e),
    # P5=(a+b)h, P6=(c-a)(e+f), P7=(b-d)(g+h)
    # C00=P1+P4-P5+P7, C01=P3+P5, C10=P2+P4, C11=P1-P2+P3+P6

    sA = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    sB = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    # W contributions per product: [C00, C01, C10, C11]
    wC = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]

    for blk in range(5):
        for r in range(7):
            c = blk * 7 + r
            for i in range(4):
                U[i][c] = sA[r][i]
            k0 = 2 * blk
            k1 = 2 * blk + 1
            V[k0][c] = sB[r][0]
            V[k1][c] = sB[r][1]
            V[p + k0][c] = sB[r][2]
            V[p + k1][c] = sB[r][3]
            W[k0 * n][c] = wC[r][0]
            W[k1 * n][c] = wC[r][1]
            W[k0 * n + 1][c] = wC[r][2]
            W[k1 * n + 1][c] = wC[r][3]

    return {"U": U, "V": V, "W": W}
