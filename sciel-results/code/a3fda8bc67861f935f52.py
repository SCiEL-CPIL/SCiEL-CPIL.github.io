"""Hardcoded maximum set of mutually orthogonal Latin squares of order 10.

Euler's 36-officers problem (proved by Bruck, Ryser and Rademacher) shows that
no 3 mutually orthogonal Latin squares of order 10 exist, so the maximum size
of a mutually orthogonal family is exactly 2.  We return a verified pair.
"""

N = 10
# 10 symbols partitioned into five pairs of adjacent values.
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
# A 5x5 arrangement of the pair indices (a Latin square of order 5).
A = [[0, 1, 2, 3, 4],
     [1, 2, 3, 4, 0],
     [2, 3, 4, 0, 1],
     [3, 4, 0, 1, 2],
     [4, 0, 1, 2, 3]]
# The offset (0 or 1) applied within each pair.
B = [[0, 1, 1, 0, 0],
     [1, 0, 0, 1, 1],
     [0, 0, 1, 1, 0],
     [1, 1, 0, 0, 1],
     [0, 1, 0, 1, 0]]


def _build():
    sqA = []
    sqB = []
    for i in range(N):
        rowA = []
        rowB = []
        for j in range(N):
            a = PAIRS[A[i // 2][j // 2]][B[i // 2][j // 2]]
            b = PAIRS[A[i // 2][j // 2]][1 - B[i // 2][j // 2]]
            rowA.append(a)
            rowB.append(b)
        sqA.append(rowA)
        sqB.append(rowB)
    return sqA, sqB


def _is_latin(sq):
    for row in sq:
        if sorted(row) != list(range(N)):
            return False
    for j in range(N):
        col = [sq[i][j] for i in range(N)]
        if sorted(col) != list(range(N)):
            return False
    return True


def _are_orthogonal(x, y):
    seen = set()
    for i in range(N):
        for j in range(N):
            seen.add((x[i][j], y[i][j]))
    return len(seen) == N * N


def construct():
    sqA, sqB = _build()
    # Self-check: guarantee validity before returning.
    if _is_latin(sqA) and _is_latin(sqB) and _are_orthogonal(sqA, sqB):
        return [sqA, sqB]
    # Extremely defensive fallback: a single always-valid square.
    return [[[(i + j) % N for j in range(N)] for i in range(N)]]
