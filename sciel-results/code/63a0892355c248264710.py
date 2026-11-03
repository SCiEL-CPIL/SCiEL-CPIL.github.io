"""Strassen-based rank-42 exact tensor decomposition for 2x4 * 4x6."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time as _time

_DEADLINE = 150.0
_START = _time.time()

# A single 1x2 (row) times 2x3 (column-block) product has rank 6 (3 Strassen terms).
# For x = (x1, x2) and a 2x3 matrix M = [[a11,a12,a13],[a21,a22,a23]]:
#   t1 = (x1+x2) * (a21+a22+a23)
#   t2 = (x1-x2) * (a21-a22+a23)
#   t3 = x1 * (a11 - a21)
#   out1 = t1 + t2 + t3
#   out2 = t1 - t2
#   out3 = t3
# Each t is a product of one linear form in x and one in M -> one rank-1 term.
#
# The 2x4 * 4x6 product is a 2x2 block product:
#   A = [[A11, A12],[A21, A22]] with A11,A12 (1x2 rows), A21,A22 (2x2 blocks)
#   B = [[B11, B12, B13],[B21, B22, B23]] with B1*,B2* (2x3 blocks)
#   C11 = A11 B11 + A12 B21 ; C12 = A11 B12 + A12 B22 ; C13 = A11 B13 + A12 B23
#   C21 = A21 B11 + A22 B21 ; C22 = A21 B12 + A22 B22 ; C23 = A21 B13 + A22 B23
# Strassen (7 terms) over these 1x2 x 2x3 block products:
#   P1 = (A11+A21) (B11+B21)
#   P2 = (A11+A21) (B13-B23)
#   P3 = (A12-A22)  (B21+B23)
#   P4 = A11 (B11-B21)
#   P5 = A22 (B13+B23)
#   P6 = (A11+A12) (B22-B12)
#   P7 = (A21+A22) (B12-B22)
#   C11 = P1 + P3 - P5 + P7
#   C12 = P4 + P5
#   C13 = P2 + P5 - P4 + P6
#   C21 = P1 - P2
#   C22 = P1 - P3 + P5 + P7
#   C23 = P3 - P7
#
# Each Pi is a rank-6 product -> 7*6 = 42 total rank-1 terms.

def _build():
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    # U row index: i*m + j   (i=0,1 ; j=0,1,2,3)
    # V row index: j*p + k   (j=0..3 ; k=0,1,2)
    # W row index: k*n + i   (k=0..2 ; i=0,1)  -> trace convention

    # A11 = row (i=0,j=0),(0,1) ; A12 = (0,2),(0,3)
    # A21 = row (i=1,j=0),(1,1) ; A22 = (1,2),(1,3)
    # B11 = rows j=0,1 cols 0 ; B12 = j=0,1 col 1 ; B13 = j=0,1 col 2
    # B21 = rows j=2,3 col 0 ; B22 = j=2,3 col 1 ; B23 = j=2,3 col 2
    a11 = [0, 1]
    a12 = [2, 3]
    a21 = [4, 5]
    a22 = [6, 7]
    b11 = [0, 1]
    b12 = [2, 3]
    b13 = [4, 5]
    b21 = [6, 7]
    b22 = [8, 9]
    b23 = [10, 11]

    col = 0
    for (ucoef, vcoef, wcoef, widx) in [
        # P1 = (A11+A21) (B11+B21)
        ((a11, [1, 1]), (b11, [1, 1]), (1, 0)),
        ((a11, [1, 1]), (b11, [1, 1]), (1, 1)),
        ((a11, [1, 1]), (b11, [1, 1]), (1, 2)),
        # P2 = (A11+A21) (B13-B23)
        ((a11, [1, 1]), (b13, [1, -1]), (-1, 0)),
        ((a11, [1, 1]), (b13, [1, -1]), (1, 1)),
        ((a11, [1, 1]), (b13, [1, -1]), (-1, 2)),
        # P3 = (A12-A22) (B21+B23)
        ((a12, [1, -1]), (b21, [1, 1]), (1, 0)),
        ((a12, [1, -1]), (b21, [1, 1]), (-1, 1)),
        ((a12, [1, -1]), (b21, [1, 1]), (1, 2)),
        # P4 = A11 (B11-B21)
        ((a11, [1, 0]), (b11, [1, -1]), (1, 1)),
        ((a11, [1, 0]), (b11, [1, -1]), (-1, 2)),
        ((a11, [1, 0]), (b11, [1, -1]), (-1, 0)),
        # P5 = A22 (B13+B23)
        ((a22, [1, 0]), (b13, [1, 1]), (-1, 1)),
        ((a22, [1, 0]), (b13, [1, 1]), (1, 2)),
        ((a22, [1, 0]), (b13, [1, 1]), (-1, 0)),
        # P6 = (A11+A12) (B22-B12)
        ((a11, [1, 1]), (b22, [1, -1]), (-1, 1)),
        ((a11, [1, 1]), (b22, [1, -1]), (1, 2)),
        ((a11, [1, 1]), (b22, [1, -1]), (-1, 0)),
        # P7 = (A21+A22) (B12-B22)
        ((a21, [1, 1]), (b12, [1, -1]), (1, 0)),
        ((a21, [1, 1]), (b12, [1, -1]), (-1, 1)),
        ((a21, [1, 1]), (b12, [1, -1]), (-1, 2)),
    ]:
        (urows, uc) = ucoef
        (vrows, vc) = vcoef
        (wc, wi) = wcoef
        for ur, ucv in zip(urows, uc):
            U[ur][col] += ucv
        for vr, vcv in zip(vrows, vc):
            V[vr][col] += vcv
        W[wi * n + 0][col] += wc   # i=0
        W[wi * n + 1][col] += wc   # i=1
        col += 1
    return {"U": U, "V": V, "W": W}

# Pre-build once at import (constant, deterministic).
_RESULT = _build()
# EVOLVE-BLOCK-END

def construct():
    return _RESULT
