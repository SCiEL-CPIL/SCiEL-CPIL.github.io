import time
import numpy as np

DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 8
    start_time = time.time()
    deadline = start_time + 110.0

    # Rank-28 decomposition: Strassen on each 2x2 column-block of B
    # 4 blocks x 7 Strassen multiplications = 28 rank-1 terms
    num_blocks = p // 2  # 4
    rank = num_blocks * 7  # 28

    U = np.zeros((n * m, rank), dtype=float)  # 4 x 28
    V = np.zeros((m * p, rank), dtype=float)  # 16 x 28
    W = np.zeros((p * n, rank), dtype=float)  # 16 x 28

    # Strassen coefficients for 2x2 x 2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6

    for b in range(num_blocks):
        base = b * 7
        k0 = 2 * b      # first column index in this block
        k1 = 2 * b + 1  # second column index in this block

        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        U[0, base + 0] += 1   # A[0,0]
        U[3, base + 0] += 1   # A[1,1]
        V[0 * p + k0, base + 0] += 1   # B[0,k0]
        V[1 * p + k1, base + 0] += 1   # B[1,k1]
        # C[0,k0] += M1, C[1,k1] += M1
        W[k0 * n + 0, base + 0] += 1
        W[k1 * n + 1, base + 0] += 1

        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        U[2, base + 1] += 1   # A[1,0]
        U[3, base + 1] += 1   # A[1,1]
        V[0 * p + k0, base + 1] += 1   # B[0,k0]
        # C[1,k0] += M2, C[1,k1] -= M2
        W[k0 * n + 1, base + 1] += 1
        W[k1 * n + 1, base + 1] -= 1

        # M3 = A[0,0] * (B[0,k1]-B[1,k1])
        U[0, base + 2] += 1   # A[0,0]
        V[0 * p + k1, base + 2] += 1   # B[0,k1]
        V[1 * p + k1, base + 2] -= 1   # B[1,k1]
        # C[0,k1] += M3, C[1,k1] += M3
        W[k1 * n + 0, base + 2] += 1
        W[k1 * n + 1, base + 2] += 1

        # M4 = A[1,1] * (B[1,k0]-B[0,k0])
        U[3, base + 3] += 1   # A[1,1]
        V[1 * p + k0, base + 3] += 1   # B[1,k0]
        V[0 * p + k0, base + 3] -= 1   # B[0,k0]
        # C[0,k0] += M4, C[1,k0] += M4
        W[k0 * n + 0, base + 3] += 1
        W[k0 * n + 1, base + 3] += 1

        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        U[0, base + 4] += 1   # A[0,0]
        U[1, base + 4] += 1   # A[0,1]
        V[1 * p + k1, base + 4] += 1   # B[1,k1]
        # C[0,k0] -= M5, C[0,k1] += M5
        W[k0 * n + 0, base + 4] -= 1
        W[k1 * n + 0, base + 4] += 1

        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[0,k1])
        U[2, base + 5] += 1   # A[1,0]
        U[0, base + 5] -= 1   # A[0,0]
        V[0 * p + k0, base + 5] += 1   # B[0,k0]
        V[0 * p + k1, base + 5] += 1   # B[0,k1]
        # C[1,k1] += M6
        W[k1 * n + 1, base + 5] += 1

        # M7 = (A[0,1]-A[1,1]) * (B[1,k0]+B[1,k1])
        U[1, base + 6] += 1   # A[0,1]
        U[3, base + 6] -= 1   # A[1,1]
        V[1 * p + k0, base + 6] += 1   # B[1,k0]
        V[1 * p + k1, base + 6] += 1   # B[1,k1]
        # C[0,k0] += M7
        W[k0 * n + 0, base + 6] += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
