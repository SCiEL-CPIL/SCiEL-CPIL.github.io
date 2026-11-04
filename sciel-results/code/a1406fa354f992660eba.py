"""Matrix multiplication 2x4x7 - rank-50 decomposition using Strassen blocks.

The 2x4x7 multiplication is decomposed into:
- Six 2x2x2 Strassen blocks (rank 7 each) = 42
- One 2x4x1 direct block (rank 8) = 8
Total: rank 50
"""
DIMENSIONS = [2, 4, 7]

import time


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()

    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 8 x 50
    V = [[0] * rank for _ in range(m * p)]  # 28 x 50
    W = [[0] * rank for _ in range(p * n)]  # 14 x 50

    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # Strassen 2x2x2 block: A[2x2] * B[2x2] -> C[2x2]
        # M1: (A00+A11)*(B00+B11)
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # M2: (A10+A11)*B00
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1

        # M3: A00*(B01-B11)
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # M4: A11*(B10-B00)
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1

        # M5: (A00+A01)*B11
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1

        # M6: (A10-A00)*(B00+B01)
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1

        # M7: (A01-A11)*(B10+B11)
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # Six 2x2x2 Strassen blocks:
    # k in {0,1}, {2,3}, {4,5}; j in {0,1}, {2,3}
    for k0 in (0, 2, 4):
        for j0 in (0, 2):
            add_strassen(0, j0, k0)

    # One 2x4x1 direct block for k=6: rank 8
    for j in range(4):
        for i in range(2):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1

    # Safety margin: return well before 120s limit
    if time.time() - start_time > 100:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
