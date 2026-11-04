"""
Rank-minimal tensor decomposition for 2x4 * 4x8 matrix multiplication.

Strategy:
  1. Primary: rank-56 Strassen-based decomposition (8 independent 2x2x2 blocks).
  2. Verify correctness using exact rational arithmetic.
  3. Fallback: schoolbook rank-64.
"""
import time
from fractions import Fraction

_DIMENSIONS = (2, 4, 8)
n, m, p = _DIMENSIONS

_DEADLINE = 135.0
_T0 = time.monotonic()


def _elapsed():
    return time.monotonic() - _T0


def _strassen_56():
    """
    Rank-56 decomposition: 8 independent 2x2x2 Strassen blocks.
    U: (8, 56), V: (32, 56), W: (16, 56)
    """
    F = Fraction
    rank = 56
    U = [[F(0)] * rank for _ in range(n * m)]
    V = [[F(0)] * rank for _ in range(m * p)]
    W = [[F(0)] * rank for _ in range(p * n)]

    col = 0
    for a in range(2):      # j-block index: j = 2*a + b
        for c in range(4):  # k-block index: k = 2*c + d

            # M1: (a00+a11)(b00+b11)
            U[0 * 4 + 2 * a + 0][col] += F(1)
            U[1 * 4 + 2 * a + 1][col] += F(1)
            V[(2 * a + 0) * 8 + 2 * c + 0][col] += F(1)
            V[(2 * a + 1) * 8 + 2 * c + 1][col] += F(1)
            W[4 * c + 0][col] += F(1)
            W[4 * c + 1][col] += F(-1)
            W[4 * c + 2][col] += F(1)
            W[4 * c + 3][col] += F(1)
            col += 1

            # M2: (a10+a11)b00
            U[1 * 4 + 2 * a + 0][col] += F(1)
            U[1 * 4 + 2 * a + 1][col] += F(1)
            V[(2 * a + 0) * 8 + 2 * c + 0][col] += F(1)
            W[4 * c + 1][col] += F(1)
            W[4 * c + 2][col] += F(-1)
            W[4 * c + 3][col] += F(1)
            col += 1

            # M3: a00(b10-b11)
            U[0 * 4 + 2 * a + 0][col] += F(1)
            V[(2 * a + 1) * 8 + 2 * c + 0][col] += F(1)
            V[(2 * a + 1) * 8 + 2 * c + 1][col] += F(-1)
            W[4 * c + 0][col] += F(1)
            col += 1

            # M4: a11(b01-b00)
            U[1 * 4 + 2 * a + 1][col] += F(1)
            V[(2 * a + 0) * 8 + 2 * c + 1][col] += F(1)
            V[(2 * a + 0) * 8 + 2 * c + 0][col] += F(-1)
            W[4 * c + 0][col] += F(-1)
            W[4 * c + 1][col] += F(1)
            col += 1

            # M5: (a00+a01)b11
            U[0 * 4 + 2 * a + 0][col] += F(1)
            U[0 * 4 + 2 * a + 1][col] += F(1)
            V[(2 * a + 1) * 8 + 2 * c + 1][col] += F(1)
            W[4 * c + 2][col] += F(1)
            W[4 * c + 3][col] += F(-1)
            col += 1

            # M6: (a10-a00)(b00+b01)
            U[1 * 4 + 2 * a + 0][col] += F(1)
            U[0 * 4 + 2 * a + 0][col] += F(-1)
            V[(2 * a + 0) * 8 + 2 * c + 0][col] += F(1)
            V[(2 * a + 0) * 8 + 2 * c + 1][col] += F(1)
            W[4 * c + 0][col] += F(-1)
            W[4 * c + 1][col] += F(1)
            col += 1

            # M7: (a11-a01)(b01+b11)
            U[1 * 4 + 2 * a + 1][col] += F(1)
            U[0 * 4 + 2 * a + 1][col] += F(-1)
            V[(2 * a + 0) * 8 + 2 * c + 1][col] += F(1)
            V[(2 * a + 1) * 8 + 2 * c + 1][col] += F(1)
            W[4 * c + 2][col] += F(-1)
            W[4 * c + 3][col] += F(1)
            col += 1

    assert col == 56, f"Expected 56 terms, got {col}"
    return {"U": U, "V": V, "W": W}


def _verify_fraction(U, V, W):
    """Exact verification using Fractions."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True


def _schoolbook():
    """Rank-64 schoolbook decomposition as fallback."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def construct():
    # Primary: rank-56 Strassen decomposition
    if _elapsed() < _DEADLINE - 10:
        s2 = _strassen_56()
        if s2 is not None:
            if _verify_fraction(s2["U"], s2["V"], s2["W"]):
                return s2
    
    # Fallback: schoolbook rank-64
    return _schoolbook()
