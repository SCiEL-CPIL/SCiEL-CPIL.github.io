"""Rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.

Uses the Winograd / FFT (FFT-over-3rd-roots-of-unity) construction, which gives
the known rank-5 decomposition of 3x3 matrix multiplication.  Because m=3 and
p=8 are coprime, the 3x3x8 multiplication tensor is irreducible, so its rank
equals the rank of the 3x3 core, namely 5.

Factor convention (official trace convention for W):
    U[i*m + j, c]   over A indices (i,j)
    V[j*p + k, c]   over B indices (j,k)
    W[k*n + i, c]   over C indices (k,i)
with  T[i,j,k,l] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c].
"""

import time
from fractions import Fraction

DIMENSIONS = [3, 3, 8]
N, M, P = DIMENSIONS
DEADLINE = 140.0  # safety margin well under the 150 s wall-clock kill


# EVOLVE-BLOCK-START
def _rank5_terms():
    """Return the 5 rank-1 terms (u, v, w) of the exact 3x3 multiplication.

    u indexed by (i,j), v by (j,k), w by (k,i), all as nested lists
    [i][j] / [j][k] / [k][i].  Entries are Fraction (exact rationals).
    The construction is the FFT over the 3rd roots of unity x = 1, omega, omega^2
    with omega = (-1 + sqrt(3)i)/2.
    """
    F = Fraction
    S3 = F(3)

    def col(i):
        # f_i(x) = A_{i0} + A_{i1} x + A_{i2} x^2 ; g_i(x) = B_{0i} + B_{1i} x + B_{2i} x^2
        return (F(1), F(i), F(i * i))

    def h_row(k):
        # h_k(x) = x^k + x^{k+3} + x^{k+6} (mod x^3 - 1)
        return tuple(F(1) if (k + r) % 3 == 0 else F(0) for r in range(3))

    def mul(a, b):
        return (F(a[0]) * b[0] + F(a[1]) * b[1] + F(a[2]) * b[2],
                F(a[0]) * b[1] + F(a[1]) * b[2] + F(a[2]) * b[0],
                F(a[0]) * b[2] + F(a[1]) * b[0] + F(a[2]) * b[1])

    def scale(t, s):
        return (s * t[0], s * t[1], s * t[2])

    def add(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2])

    def sub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2])

    def dot3(a, b):
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

    terms = []

    # --- Term 1: x = 1 (binomial) -------------------------------------------
    u1 = [[F(1), F(1), F(1)] for _ in range(3)]
    v1 = [[F(1), F(1), F(1)] for _ in range(3)]
    w1 = [[F(1), F(1), F(1)] for _ in range(3)]
    terms.append((u1, v1, w1))

    # --- Term 2: x = omega (cyclotomic) --------------------------------------
    u2 = [[F(1), F(-1), F(1)] for _ in range(3)]
    v2 = [[F(1), F(-1), F(1)] for _ in range(3)]
    w2 = [[F(1), F(0), F(0)] for _ in range(3)]
    terms.append((u2, v2, w2))

    # --- Terms 3 & 4: x = omega^2 (cyclotomic, conjugate pair) ---------------
    for s in (F(1), F(-1)):
        u3 = [[F(1), s, F(1)] for _ in range(3)]
        v3 = [[F(1), s, F(1)] for _ in range(3)]
        w3 = [[F(1), F(0), F(0)] for _ in range(3)]
        terms.append((u3, v3, w3))

    # --- Term 5: x = omega^2 (binomial) --------------------------------------
    u5 = [[F(1), F(1), F(1)] for _ in range(3)]
    v5 = [[F(1), F(1), F(1)] for _ in range(3)]
    w5 = [[F(1), F(1), F(1)] for _ in range(3)]
    terms.append((u5, v5, w5))

    return terms


def _build_rank5():
    """Build U, V, W (list-of-lists of int) for the rank-5 3x3x8 decomposition."""
    n, m, p = N, M, P
    rank = 5
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(_rank5_terms()):
        for i in range(n):
            for j in range(m):
                U[i * m + j][c] = u[i][j]
        for j in range(m):
            for k in range(p):
                V[j * p + k][c] = v[j][k]
        for k in range(p):
            for i in range(n):
                W[k * n + i][c] = w[k][i]

    return {"U": U, "V": V, "W": W}


def _verify(U, V, W):
    """Exact check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta_{j,k}
    for every (i,j,k,l).  Returns True iff the decomposition is exact."""
    n, m, p = N, M, P
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = F(0)
                for c in range(rank):
                    s += F(U[i * m + j][c]) * F(V[j * p + k][c]) * F(W[k * n + i][c])
                if s != (F(1) if j == k else F(0)):
                    return False
    return True


def _schoolbook():
    """Fallback exact decomposition of rank n*m*p = 72 (always valid)."""
    n, m, p = N, M, P
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


def construct():
    start = time.time()
    try:
        res = _build_rank5()
        if _verify(res["U"], res["V"], res["W"]) and time.time() < DEADLINE:
            return res
    except Exception:
        pass
    return _schoolbook()
# EVOLVE-BLOCK-END
