"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction
from math import gcd

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _build_decomposition():
    n, m, p = 3, 5, 7
    # U: 15 rows (A entries), V: 35 rows (B entries), W: 21 rows (C entries)
    U = {}  # (i,j) -> list of (col, coeff)
    V = {}  # (j,k) -> list of (col, coeff)
    W = {}  # (k,i) -> list of (col, coeff)
    col = 0

    def add_U(i, j, c):
        U.setdefault((i, j), []).append((col, c))

    def add_V(j, k, c):
        V.setdefault((j, k), []).append((col, c))

    def add_W(k, i, c):
        W.setdefault((k, i), []).append((col, c))

    def strassen(a00, a01, a10, a11, b00, b01, b10, b11, c00, c01, c10, c11):
        # returns global column of M7 (the last term), which is added to c11
        nonlocal col
        M1 = (a00, a11, b01, b11, 1, -1, c00, 1)
        M2 = (a10, a11, b00, None, 1, 1, c10, 1)
        M3 = (a00, None, b10, b11, 1, -1, c01, 1)
        M4 = (a11, None, b00, b10, 1, -1, c10, 1)
        M5 = (a00, a01, b11, None, 1, 1, c00, -1)
        M6 = (a10, a00, b01, None, 1, -1, c11, 1)
        M7 = (a01, a11, b10, b11, 1, 1, c00, 1)

        def emit(term):
            nonlocal col
            (x0, x1, y0, y1, s0, s1, cpos, sC) = term
            add_U(x0[0], x0[1], s0)
            if x1 is not None:
                add_U(x1[0], x1[1], s1)
            add_V(y0[0], y0[1], s0)
            if y1 is not None:
                add_V(y1[0], y1[1], s1)
            add_W(cpos[1], cpos[0], sC)
            col += 1

        for term in (M1, M2, M3, M4, M5, M6, M7):
            emit(term)
        return M7

    # Block 1: A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    strassen((0, 0), (0, 1), (1, 0), (1, 1),
             (0, 0), (0, 1), (1, 0), (1, 1),
             (0, 0), (1, 0), (0, 1), (1, 1))

    # Block 2: A[0:2,2:4] * B[2:4,2:4] -> C[0:2,2:4]
    strassen((0, 2), (0, 3), (1, 2), (1, 3),
             (2, 2), (2, 3), (3, 2), (3, 3),
             (2, 0), (3, 0), (2, 1), (3, 1))

    # Block 3: A[1:3,2:4] * B[2:4,2:4] -> C[1:3,2:4]
    strassen((1, 2), (1, 3), (2, 2), (2, 3),
             (2, 2), (2, 3), (3, 2), (3, 3),
             (2, 2), (3, 2), (2, 3), (3, 3))

    # Schoolbook terms (7)
    def schoolbook(i, j, k, c):
        nonlocal col
        add_U(i, j, 1)
        add_V(j, k, 1)
        add_W(k, i, 1)
        col += 1

    # A[0,4]*B[4,0:7], A[1,4]*B[4,0:7], A[2,4]*B[4,0:7]
    for k in range(7):
        schoolbook(0, 4, k, 1)
    for k in range(7):
        schoolbook(1, 4, k, 1)
    for k in range(7):
        schoolbook(2, 4, k, 1)
    # A[0:3,2]*B[2,0:2], A[0:3,3]*B[3,0:2]
    for k in range(2):
        schoolbook(0, 2, k, 1)
        schoolbook(1, 2, k, 1)
        schoolbook(2, 2, k, 1)
    for k in range(2):
        schoolbook(0, 3, k, 1)
        schoolbook(1, 3, k, 1)
        schoolbook(2, 3, k, 1)
    # A[0:3,0]*B[0,4:7], A[0:3,1]*B[1,4:7]
    for k in range(4, 7):
        schoolbook(0, 0, k, 1)
        schoolbook(1, 0, k, 1)
        schoolbook(2, 0, k, 1)
    for k in range(4, 7):
        schoolbook(0, 1, k, 1)
        schoolbook(1, 1, k, 1)
        schoolbook(2, 1, k, 1)

    rank = col
    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    for (i, j), lst in U.items():
        for c, v in lst:
            U_mat[i * m + j][c] = v
    for (j, k), lst in V.items():
        for c, v in lst:
            V_mat[j * p + k][c] = v
    for (k, i), lst in W.items():
        for c, v in lst:
            W_mat[k * n + i][c] = v
    return U_mat, V_mat, W_mat, rank


def _verify(U, V, W):
    n, m, p = 3, 5, 7
    rank = len(U[0])
    for i in range(n):
        for k in range(p):
            s = Fraction(0)
            for j in range(m):
                for r in range(rank):
                    s += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r])
            target = Fraction(1) if i == k else Fraction(0)
            if s != target:
                return False
    return True


def _to_ints(U, V, W):
    # Find LCM of all denominators, scale to integers.
    den = 1
    for M in (U, V, W):
        for row in M:
            for x in row:
                d = x.denominator
                den = den * d // gcd(den, d)
    scale = den
    def conv(M):
        return [[x.numerator * scale // x.denominator for x in row] for row in M]
    return conv(U), conv(V), conv(W)


_CACHED = None


def _get_result():
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    U, V, W, rank = _build_decomposition()
    if _verify(U, V, W):
        Ui, Vi, Wi = _to_ints(U, V, W)
        _CACHED = {"U": Ui, "V": Vi, "W": Wi}
        return _CACHED
    # Fallback: schoolbook rank 105
    n, m, p = 3, 5, 7
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
    _CACHED = {"U": U, "V": V, "W": W}
    return _CACHED


def construct():
    deadline = time.time() + 130.0
    res = _get_result()
    return res
# EVOLVE-BLOCK-END
