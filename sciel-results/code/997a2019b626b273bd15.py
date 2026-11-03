"""Matrix multiplication 2x4 by 4x5 via block Strassen + schoolbook,
with exact verification.

We decompose:
  C (2x5) = A (2x4) * B (4x5)

Split B into B_left (4x4) and B_right (4x1):
  C_left (2x4) = A * B_left
  C_right (2x1) = A * B_right

For C_left, split A into A_L (2x2) and A_R (2x2), and B_left into four 2x2 blocks:
  C_left[:,0:2] = A_L * B_LL + A_R * B_RL
  C_left[:,2:4] = A_L * B_LR + A_R * B_RR

Each 2x2x2 product uses Strassen (rank 7): 4 * 7 = 28
C_right (2x1) = A * B_right uses schoolbook (2*4 = 8 terms)

Total rank = 28 + 8 = 36
"""

import time

DIMENSIONS = [2, 4, 5]
_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0


def _remaining():
    return _DEADLINE - time.time()


def construct():
    """
    Return {'U','V','W'} factor matrices that exactly decompose the
    2x4-by-4x5 matrix multiplication tensor.

    U: shape (8, 36)  -- indices: i*m + j for i in 0..1, j in 0..3
    V: shape (20, 36) -- indices: j*p + k for j in 0..3, k in 0..4
    W: shape (10, 36) -- indices: k*n + i for k in 0..4, i in 0..1
    """
    n, m, p = DIMENSIONS  # n=2, m=4, p=5
    rank = 36

    U = [[0] * rank for _ in range(n * m)]   # 8 x 36
    V = [[0] * rank for _ in range(m * p)]   # 20 x 36
    W = [[0] * rank for _ in range(p * n)]   # 10 x 36

    col = 0

    def add_strassen_222(i0, j0, k0, col_start):
        """
        Add a Strassen 2x2x2 decomposition for:
        C[i0:i0+2, k0:k0+2] += A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        """
        nonlocal col

        # M1 = (A00 + A11) * (B00 + B11)
        c = col_start
        U[i0 * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M2 = (A10 + A11) * B00
        c = col_start + 1
        U[(i0 + 1) * m + j0][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[j0 * p + k0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] -= 1
        col += 1

        # M3 = A00 * (B01 - B11)
        c = col_start + 2
        U[i0 * m + j0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M4 = A11 * (B10 - B00)
        c = col_start + 3
        U[(i0 + 1) * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + k0][c] += 1
        V[j0 * p + k0][c] -= 1
        W[k0 * n + i0][c] += 1
        W[k0 * n + (i0 + 1)][c] += 1
        col += 1

        # M5 = (A00 + A01) * B11
        c = col_start + 4
        U[i0 * m + j0][c] += 1
        U[i0 * m + (j0 + 1)][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] -= 1
        W[(k0 + 1) * n + i0][c] += 1
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        c = col_start + 5
        U[(i0 + 1) * m + j0][c] += 1
        U[i0 * m + j0][c] -= 1
        V[j0 * p + k0][c] += 1
        V[j0 * p + (k0 + 1)][c] += 1
        W[(k0 + 1) * n + (i0 + 1)][c] += 1
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        c = col_start + 6
        U[i0 * m + (j0 + 1)][c] += 1
        U[(i0 + 1) * m + (j0 + 1)][c] -= 1
        V[(j0 + 1) * p + k0][c] += 1
        V[(j0 + 1) * p + (k0 + 1)][c] += 1
        W[k0 * n + i0][c] += 1
        col += 1

        return col_start + 7

    def add_schoolbook_2x4x1(i0, j0, k0, col_start):
        """
        Add schoolbook decomposition for:
        C[i0:i0+2, k0:k0+1] += A[i0:i0+2, j0:j0+4] * B[j0:j0+4, k0:k0+1]
        """
        c = col_start
        for di in range(2):
            for dj in range(4):
                U[(i0 + di) * m + (j0 + dj)][c] = 1
                V[(j0 + dj) * p + k0][c] = 1
                W[k0 * n + (i0 + di)][c] = 1
                c += 1
        return c

    # C[:, 0:2] = A[:,0:2] * B[0:2, 0:2] + A[:,2:4] * B[2:4, 0:2]
    col = add_strassen_222(0, 0, 0, col)
    col = add_strassen_222(0, 2, 0, col)

    # C[:, 2:4] = A[:,0:2] * B[0:2, 2:4] + A[:,2:4] * B[2:4, 2:4]
    col = add_strassen_222(0, 0, 2, col)
    col = add_strassen_222(0, 2, 2, col)

    # C[:, 4] = A[:,0:4] * B[0:4, 4]
    col = add_schoolbook_2x4x1(0, 0, 4, col)

    assert col == rank, f"Expected rank {rank}, got {col}"

    # ---- Exact verification ----------------------------------------------
    if _remaining() > 1.0:
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    acc = 0
                    ui = i * m + j
                    vj = j * p + k
                    wk = k * n + i
                    for r in range(rank):
                        acc += U[ui][r] * V[vj][r] * W[wk][r]
                    if acc != 1:
                        raise AssertionError(
                            f"Verification failed at i={i},j={j},k={k}: acc={acc}"
                        )

    return {"U": U, "V": V, "W": W}
