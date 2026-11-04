import numpy as np
import time

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 118.0
_CONSTRUCTED = None

def construct():
    global _CONSTRUCTED
    if _CONSTRUCTED is not None:
        return _CONSTRUCTED

    # 4x4 matrix multiplication via Kronecker product of 2x2 Strassen.
    # Rank = 7 * 7 = 49.
    #
    # 2x2 Strassen factors (rank 7):
    #   M0 = (a11+a22)*(b11+b22)
    #   M1 = (a21+a22)*b11
    #   M2 = a11*(b12-b22)
    #   M3 = a22*(b21-b11)
    #   M4 = (a11+a12)*b22
    #   M5 = (a21-a11)*(b11+b12)
    #   M6 = (a12-a22)*(b21+b22)
    #
    # C11 = M0 - M1 + M3 + M5
    # C12 = M2 + M4
    # C21 = M1 + M6
    # C22 = M0 - M2 + M3 + M4

    # U2[row_idx, k] where row_idx = i*2+j for A_ij (i,j in {0,1})
    # Row order: a11, a12, a21, a22
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=np.int8)

    # V2[row_idx, k] where row_idx = j*2+k for B_jk (j,k in {0,1})
    # Row order: b11, b12, b21, b22
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=np.int8)

    # W2[row_idx, k] where row_idx = k*2+i for C_ki (k,i in {0,1})
    # Row order: c11, c12, c21, c22
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=np.int8)

    R = 49
    U = np.zeros((16, R), dtype=np.int8)
    V = np.zeros((16, R), dtype=np.int8)
    W = np.zeros((16, R), dtype=np.int8)

    for a in range(4):
        for b in range(4):
            U[a * 4 + b] = np.outer(U2[a, :], U2[b, :]).ravel()

    for b in range(4):
        for c in range(4):
            V[b * 4 + c] = np.outer(V2[b, :], V2[c, :]).ravel()

    for c in range(4):
        for a in range(4):
            W[c * 4 + a] = np.outer(W2[c, :], W2[a, :]).ravel()

    _CONSTRUCTED = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return _CONSTRUCTED
# EVOLVE-BLOCK-END
