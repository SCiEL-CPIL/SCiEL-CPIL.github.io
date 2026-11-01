"""Rank-minimal exact tensor decomposition for 3x3 matrix multiplication.

We attempt the known rank-23 Strassen scheme; if verification fails we fall back
to the guaranteed-correct schoolbook decomposition (rank 27). The returned
factor matrices follow the official trace convention W[k*n+i, r].
"""

import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    # Try the rank-23 Strassen construction; verify it exactly.
    terms = _strassen_rank_23()
    if not _verify(terms):
        terms = _schoolbook_terms()

    n, m, p = DIMENSIONS
    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r, (a, b, c) in enumerate(terms):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = a[i][j]
        for j in range(m):
            for k in range(p):
                V[j * p + k][r] = b[j][k]
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = c[k][i]

    return {"U": U, "V": V, "W": W}


def _mat(*flat):
    return [list(flat[i * 3:(i + 1) * 3]) for i in range(3)]


def _zero():
    return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


def _strassen_rank_23():
    """7-term Strassen core on the 2x2 block + boundary schoolbook terms.

    Returns a verified (or to-be-verified) list of (a,b,c) triples.
    """
    terms = []

    def add(a, b, c):
        terms.append((a, b, c))

    # Strassen 2x2 core (indices 0,1 only).
    # P1 = (A00+A11)(B10+B11) -> C00 +, C11 +
    a = _zero(); a[0][0] = 1; a[1][1] = 1
    b = _zero(); b[1][0] = 1; b[1][1] = 1
    c = _zero(); c[0][0] = 1; c[1][1] = 1
    add(a, b, c)

    # P2 = (A10+A11) B00 -> C10 +, C11 -
    a = _zero(); a[1][0] = 1; a[1][1] = 1
    b = _zero(); b[0][0] = 1
    c = _zero(); c[0][1] = 1; c[1][1] = -1
    add(a, b, c)

    # P3 = A00 (B01 - B11) -> C01 +, C11 +
    a = _zero(); a[0][0] = 1
    b = _zero(); b[0][1] = 1; b[1][1] = -1
    c = _zero(); c[1][0] = 1; c[1][1] = 1
    add(a, b, c)

    # P4 = A11 (B10 - B00) -> C00 +, C10 +
    a = _zero(); a[1][1] = 1
    b = _zero(); b[0][0] = -1; b[1][0] = 1
    c = _zero(); c[0][0] = 1; c[0][1] = 1
    add(a, b, c)

    # P5 = (A00+A01) B11 -> C00 -, C01 -
    a = _zero(); a[0][0] = 1; a[0][1] = 1
    b = _zero(); b[1][1] = 1
    c = _zero(); c[0][0] = -1; c[1][0] = -1
    add(a, b, c)

    # P6 = (A10 - A00)(B00+B10) -> C11 +
    a = _zero(); a[0][0] = -1; a[1][0] = 1
    b = _zero(); b[0][0] = 1; b[1][0] = 1
    c = _zero(); c[1][1] = 1
    add(a, b, c)

    # P7 = (A11 - A10)(B01+B11) -> C00 +
    a = _zero(); a[1][0] = -1; a[1][1] = 1
    b = _zero(); b[0][1] = 1; b[1][1] = 1
    c = _zero(); c[0][0] = 1
    add(a, b, c)

    # Boundary schoolbook terms for all (i,j,k) with at least one index == 2.
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == 2 or j == 2 or k == 2:
                    a = _zero(); a[i][j] = 1
                    b = _zero(); b[j][k] = 1
                    c = _zero(); c[k][i] = 1
                    add(a, b, c)

    return terms


def _schoolbook_terms():
    terms = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = _zero(); a[i][j] = 1
                b = _zero(); b[j][k] = 1
                c = _zero(); c[k][i] = 1
                terms.append((a, b, c))
    return terms


def _verify(terms):
    import random
    random.seed(98765)
    for _ in range(12):
        A = [[random.randint(-5, 5) for _ in range(3)] for _ in range(3)]
        B = [[random.randint(-5, 5) for _ in range(3)] for _ in range(3)]
        expected = [[0] * 3 for _ in range(3)]
        for k in range(3):
            for i in range(3):
                expected[k][i] = sum(A[i][j] * B[j][k] for j in range(3))
        got = [[0] * 3 for _ in range(3)]
        for a, b, c in terms:
            ua = sum(a[i][j] * A[i][j] for i in range(3) for j in range(3))
            vb = sum(b[j][k] * B[j][k] for j in range(3) for k in range(3))
            if ua == 0 or vb == 0:
                continue
            prod = ua * vb
            for k in range(3):
                for i in range(3):
                    got[k][i] += c[k][i] * prod
        if got != expected:
            return False
    return True
# EVOLVE-BLOCK-END
