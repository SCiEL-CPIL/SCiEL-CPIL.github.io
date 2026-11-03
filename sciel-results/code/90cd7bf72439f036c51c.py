import numpy as np

DIMENSIONS = [4, 4, 8]

def _strassen_2x2_factors():
    """
    Returns U, V, W factors for 2x2x2 matrix multiplication with rank 7.
    U: (2, 2, 7), V: (2, 2, 7), W: (2, 2, 7)
    Indices: U[i, j, r], V[j, k, r], W[k, i, r]
    """
    U = np.zeros((2, 2, 7), dtype=float)
    V = np.zeros((2, 2, 7), dtype=float)
    W = np.zeros((2, 2, 7), dtype=float)

    # r=0: (A00+A11)(B00+B11)
    U[0, 0, 0] = 1; U[1, 1, 0] = 1
    V[0, 0, 0] = 1; V[1, 1, 0] = 1
    W[0, 0, 0] = 1; W[1, 1, 0] = 1

    # r=1: (A10+A11)B00
    U[1, 0, 1] = 1; U[1, 1, 1] = 1
    V[0, 0, 1] = 1
    W[0, 1, 1] = 1; W[1, 1, 1] = -1

    # r=2: A00(B01-B11)
    U[0, 0, 2] = 1
    V[0, 1, 2] = 1; V[1, 1, 2] = -1
    W[1, 0, 2] = 1; W[1, 1, 2] = 1

    # r=3: A11(B10-B00)
    U[1, 1, 3] = 1
    V[1, 0, 3] = 1; V[0, 0, 3] = -1
    W[0, 0, 3] = 1; W[0, 1, 3] = 1

    # r=4: (A00+A01)B11
    U[0, 0, 4] = 1; U[0, 1, 4] = 1
    V[1, 1, 4] = 1
    W[0, 0, 4] = -1; W[1, 0, 4] = 1

    # r=5: (A10-A00)(B00+B01)
    U[1, 0, 5] = 1; U[0, 0, 5] = -1
    V[0, 0, 5] = 1; V[0, 1, 5] = 1
    W[1, 1, 5] = 1

    # r=6: (A01-A11)(B10+B11)
    U[0, 1, 6] = 1; U[1, 1, 6] = -1
    V[1, 0, 6] = 1; V[1, 1, 6] = 1
    W[0, 0, 6] = 1

    return U, V, W

def _build_4x4_4x8_strassen():
    """
    Builds rank-98 decomposition for 4x4x8 matrix multiplication
    using Kronecker product of Strassen 2x2x2 factors.
    Returns U (16, 98), V (32, 98), W (32, 98).
    """
    n, m, p = 4, 4, 8
    U2, V2, W2 = _strassen_2x2_factors()
    
    # Build 4x4x4 factors (Rank 49)
    U4 = np.zeros((4, 4, 49), dtype=float)
    V4 = np.zeros((4, 4, 49), dtype=float)
    W4 = np.zeros((4, 4, 49), dtype=float)

    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2 * i1 + i0
                    j = 2 * j1 + j0
                    for s in range(7):
                        for r in range(7):
                            R = 7 * s + r
                            U4[i, j, R] = U2[i1, j1, s] * U2[i0, j0, r]

    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2 * j1 + j0
                    k = 2 * k1 + k0
                    for s in range(7):
                        for r in range(7):
                            R = 7 * s + r
                            V4[j, k, R] = V2[j1, k1, s] * V2[j0, k0, r]

    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2 * k1 + k0
                    i = 2 * i1 + i0
                    for s in range(7):
                        for r in range(7):
                            R = 7 * s + r
                            W4[k, i, R] = W2[k1, i1, s] * W2[k0, i0, r]

    # Build 4x4x8 factors (Rank 98)
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)

    # First half: B[:, 0:4]
    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i * 4 + j, R] = U4[i, j, R]

    for j in range(4):
        for k in range(4):
            for R in range(49):
                V[j * 8 + k, R] = V4[j, k, R]

    for k in range(4):
        for i in range(4):
            for R in range(49):
                W[k * 4 + i, R] = W4[k, i, R]

    # Second half: B[:, 4:8]
    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i * 4 + j, 49 + R] = U4[i, j, R]

    for j in range(4):
        for kp in range(4):
            k = 4 + kp
            for R in range(49):
                V[j * 8 + k, 49 + R] = V4[j, kp, R]

    for kp in range(4):
        k = 4 + kp
        for i in range(4):
            for R in range(49):
                W[k * 4 + i, 49 + R] = W4[kp, i, R]

    return U, V, W

def construct():
    U, V, W = _build_4x4_4x8_strassen()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
