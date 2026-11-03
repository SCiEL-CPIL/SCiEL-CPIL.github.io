import numpy as np
import time

def construct():
    n, m, p = 2, 3, 8
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    # 2x1x8 part: A[:,0] * B[0,:]
    for i in range(n):
        for k in range(p):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1

    # 2x2x8 part via Strassen on four 2x2x2 blocks
    for blk in range(4):
        c0 = blk * 2
        c1 = blk * 2 + 1
        b00, b01 = 8 + c0, 8 + c1
        b10, b11 = 16 + c0, 16 + c1
        w00, w01 = c0 * 2, c0 * 2 + 1
        w10, w11 = c1 * 2, c1 * 2 + 1

        # m1 = (a00+a11)(b00+b11)
        U[1, col] = 1.0; U[5, col] = 1.0
        V[b00, col] = 1.0; V[b11, col] = 1.0
        W[w00, col] = 1.0; W[w11, col] = 1.0
        col += 1
        # m2 = (a10+a11)b00
        U[4, col] = 1.0; U[5, col] = 1.0
        V[b00, col] = 1.0
        W[w10, col] = 1.0; W[w11, col] = -1.0
        col += 1
        # m3 = a00(b01-b11)
        U[1, col] = 1.0
        V[b01, col] = 1.0; V[b11, col] = -1.0
        W[w01, col] = 1.0; W[w11, col] = 1.0
        col += 1
        # m4 = a11(b10-b00)
        U[5, col] = 1.0
        V[b10, col] = 1.0; V[b00, col] = -1.0
        W[w00, col] = 1.0; W[w10, col] = 1.0
        col += 1
        # m5 = (a00+a01)b11
        U[1, col] = 1.0; U[2, col] = 1.0
        V[b11, col] = 1.0
        W[w00, col] = -1.0; W[w01, col] = 1.0
        col += 1
        # m6 = (a10-a00)(b00+b01)
        U[4, col] = 1.0; U[1, col] = -1.0
        V[b00, col] = 1.0; V[b01, col] = 1.0
        W[w11, col] = 1.0
        col += 1
        # m7 = (a01-a11)(b10+b11)
        U[2, col] = 1.0; U[5, col] = -1.0
        V[b10, col] = 1.0; V[b11, col] = 1.0
        W[w00, col] = 1.0
        col += 1

    U_list = [[U[i, r] for r in range(rank)] for i in range(n * m)]
    V_list = [[V[i, r] for r in range(rank)] for i in range(m * p)]
    W_list = [[W[i, r] for r in range(rank)] for i in range(p * n)]
    return {"U": U_list, "V": V_list, "W": W_list}
