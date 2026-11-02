import time
import numpy as np

def construct():
    t0 = time.time()
    n, m, p = 2, 2, 16
    nb = p // m  # 8 column-pairs in B
    r = nb * 7   # rank = 56

    U = [[0] * r for _ in range(n * m)]   # 4 x 56
    V = [[0] * r for _ in range(m * p)]   # 32 x 56
    W = [[0] * r for _ in range(p * n)]   # 32 x 56

    for t in range(nb):
        base = t * 7
        bA = 2 * t
        bB = 2 * t + 1
        bC = 16 + 2 * t
        bD = 16 + 2 * t + 1

        c00 = 4 * t
        c01 = 4 * t + 1
        c10 = 4 * t + 2
        c11 = 4 * t + 3

        # M1 = (A00+A11)*(B00+B11)
        U[0][base] = 1; U[3][base] = 1
        V[bA][base] = 1; V[bD][base] = 1
        W[c00][base] = 1; W[c11][base] = 1

        # M2 = (A10+A11)*B00
        U[2][base+1] = 1; U[3][base+1] = 1
        V[bA][base+1] = 1
        W[c01][base+1] = 1; W[c11][base+1] = -1

        # M3 = A00*(B01-B11)
        U[0][base+2] = 1
        V[bB][base+2] = 1; V[bD][base+2] = -1
        W[c10][base+2] = 1; W[c11][base+2] = 1

        # M4 = A11*(B10-B00)
        U[3][base+3] = 1
        V[bC][base+3] = 1; V[bA][base+3] = -1
        W[c00][base+3] = 1; W[c01][base+3] = 1

        # M5 = (A00+A01)*B11
        U[0][base+4] = 1; U[1][base+4] = 1
        V[bD][base+4] = 1
        W[c00][base+4] = -1; W[c10][base+4] = 1

        # M6 = (A10-A00)*(B00+B01)
        U[2][base+5] = 1; U[0][base+5] = -1
        V[bA][base+5] = 1; V[bB][base+5] = 1
        W[c11][base+5] = 1

        # M7 = (A01-A11)*(B10+B11)
        U[1][base+6] = 1; U[3][base+6] = -1
        V[bC][base+6] = 1; V[bD][base+6] = 1
        W[c00][base+6] = 1

    return {"U": U, "V": V, "W": W}
