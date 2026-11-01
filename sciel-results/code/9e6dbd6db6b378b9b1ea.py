import time

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Provably optimal rank-7 decomposition (Strassen) for 2x2 matrix multiplication.
    # Score target: seed/rank = 8/7 ≈ 1.142857...
    rank = 7

    # Allocate factor matrices
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(4)]
    W = [[0.0] * rank for _ in range(4)]

    # Fill in the 7 Strassen slices
    # Slice 0
    U[0][0] = 1.0
    U[3][0] = 1.0
    V[0][0] = 1.0
    V[3][0] = 1.0
    W[0][0] = 1.0
    W[3][0] = 1.0

    # Slice 1
    U[2][1] = 1.0
    U[3][1] = 1.0
    V[0][1] = 1.0
    W[1][1] = 1.0
    W[3][1] = -1.0

    # Slice 2
    U[0][2] = 1.0
    V[1][2] = 1.0
    V[3][2] = -1.0
    W[2][2] = 1.0
    W[3][2] = 1.0

    # Slice 3
    U[3][3] = 1.0
    V[0][3] = -1.0
    V[2][3] = 1.0
    W[0][3] = 1.0
    W[1][3] = 1.0

    # Slice 4
    U[0][4] = 1.0
    U[1][4] = 1.0
    V[3][4] = 1.0
    W[0][4] = -1.0
    W[2][4] = 1.0

    # Slice 5
    U[0][5] = -1.0
    U[2][5] = 1.0
    V[0][5] = 1.0
    V[1][5] = 1.0
    W[3][5] = 1.0

    # Slice 6
    U[1][6] = 1.0
    U[3][6] = -1.0
    V[2][6] = 1.0
    V[3][6] = 1.0
    W[0][6] = 1.0

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
