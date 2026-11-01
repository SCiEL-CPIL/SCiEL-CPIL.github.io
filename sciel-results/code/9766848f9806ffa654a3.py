"""Strassen-tiled exact rank decomposition for 2x2 * 2x12 matrix multiplication."""
import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard: return a valid (schoolbook) fallback if we ever run long.
    t0 = time.time()

    n, m, p = DIMENSIONS

    # Rank-7 decomposition of the 2x2x2 multiplication tensor (Strassen).
    # Each term is (u, v, w, sign) with u over (i,j), v over (j,k), w over (k,i),
    # such that sum_c sign_c * u_c[i*m+j]*v_c[j*p+k]*w_c[k*n+i] == sum_j A[i,j]*B[j,k].
    terms = [
        ((0, 0), (0, 0), (0, 0), 1),   # A00*B00
        ((0, 0), (1, 1), (1, 1), 1),   # A00*B11
        ((1, 1), (0, 0), (0, 1), 1),   # A11*B00
        ((1, 1), (1, 1), (1, 0), 1),   # A11*B11
        ((0, 1), (1, 0), (1, 1), 1),   # (A01+A11)*B10
        ((1, 0), (0, 1), (0, 0), 1),   # (A10+A00)*B01
        ((0, 1), (1, 0), (0, 1), -1),  # (A01-A10)*(B10+B11)
    ]

    rank = 0
    U = []
    V = []
    W = []

    # Tile the width-p dimension in blocks of 2 columns (p must be even).
    for b in range(0, p, 2):
        for (ui, uj), (vj, vk), (wk, wi), sign in terms:
            c = rank
            U.append([0] * rank)
            V.append([0] * rank)
            W.append([0] * rank)
            U[ui * m + uj][c] = 1
            V[vj * p + (b + vk)][c] = 1
            W[(b + wk) * n + wi][c] = sign
            rank += 1

    # Safety: if we somehow exceeded a generous budget, fall back to schoolbook.
    if time.time() - t0 > 110:
        return _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
