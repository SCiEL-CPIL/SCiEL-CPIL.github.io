import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # 4 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 16 x 28

    for s in range(4):
        k0 = 2 * s
        k1 = 2 * s + 1

        # V indices: V[j*8+k]
        v00 = 0 * 8 + k0   # B[0,k0]
        v01 = 0 * 8 + k1   # B[0,k1]
        v10 = 1 * 8 + k0   # B[1,k0]
        v11 = 1 * 8 + k1   # B[1,k1]

        # W indices: W[k*2+i] (official trace convention)
        w00 = k0 * 2 + 0   # C[k0, 0]
        w01 = k0 * 2 + 1   # C[k0, 1]
        w10 = k1 * 2 + 0   # C[k1, 0]
        w11 = k1 * 2 + 1   # C[k1, 1]

        # U indices: U[i*2+j]
        # A00=0, A01=1, A10=2, A11=3

        # Term 1: M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        r = s * 7 + 0
        U[0][r] += 1; U[3][r] += 1
        V[v00][r] += 1; V[v11][r] += 1
        W[w00][r] += 1; W[w11][r] += 1

        # Term 2: M2 = (A10+A11)B00 -> C10 += M2, C11 -= M2
        r = s * 7 + 1
        U[2][r] += 1; U[3][r] += 1
        V[v00][r] += 1
        W[w10][r] += 1; W[w11][r] += -1

        # Term 3: M3 = A00(B10-B00) -> C01 += M3, C11 += M3
        r = s * 7 + 2
        U[0][r] += 1
        V[v10][r] += 1; V[v00][r] += -1
        W[w01][r] += 1; W[w11][r] += 1

        # Term 4: M4 = A11(B01-B10) -> C00 += M4, C10 += M4
        r = s * 7 + 3
        U[3][r] += 1
        V[v01][r] += 1; V[v10][r] += -1
        W[w00][r] += 1; W[w10][r] += 1

        # Term 5: M5 = (A00+A01)B11 -> C00 -= M5, C01 += M5
        r = s * 7 + 4
        U[0][r] += 1; U[1][r] += 1
        V[v11][r] += 1
        W[w00][r] += -1; W[w01][r] += 1

        # Term 6: M6 = (A10-A00)(B00+B01) -> C11 += M6
        r = s * 7 + 5
        U[2][r] += 1; U[0][r] += -1
        V[v00][r] += 1; V[v01][r] += 1
        W[w11][r] += 1

        # Term 7: M7 = (A01-A10)(B10+B11) -> C00 += M7
        r = s * 7 + 6
        U[1][r] += 1; U[2][r] += -1
        V[v10][r] += 1; V[v11][r] += 1
        W[w00][r] += 1

    return {"U": U, "V": V, "W": W}
