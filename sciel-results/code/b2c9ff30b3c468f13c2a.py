"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

# Hardcoded valid Latin squares of order 10
L1 = [[(i + j) % 10 for j in range(10)] for i in range(10)]
L2 = [
    [0, 3, 6, 9, 2, 5, 8, 1, 4, 7],
    [1, 4, 7, 0, 3, 6, 9, 2, 5, 8],
    [2, 5, 8, 1, 4, 7, 0, 3, 6, 9],
    [3, 6, 9, 2, 5, 8, 1, 4, 7, 0],
    [4, 7, 0, 3, 6, 9, 2, 5, 8, 1],
    [5, 8, 1, 4, 7, 0, 3, 6, 9, 2],
    [6, 9, 2, 5, 8, 1, 4, 7, 0, 3],
    [7, 0, 3, 6, 9, 2, 5, 8, 1, 4],
    [8, 1, 4, 7, 0, 3, 6, 9, 2, 5],
    [9, 2, 5, 8, 1, 4, 7, 0, 3, 6],
]

# I will hardcode 4 more squares.
# These are known to be orthogonal to L1, L2 and each other.
L3 = [
    [0, 4, 8, 2, 6, 1, 5, 9, 3, 7],
    [1, 5, 9, 3, 7, 2, 6, 0, 4, 8],
    [2, 6, 0, 4, 8, 3, 7, 1, 5, 9],
    [3, 7, 1, 5, 9, 4, 8, 2, 6, 0],
    [4, 8, 2, 6, 0, 5, 9, 3, 7, 1],
    [5, 9, 3, 7, 1, 6, 0, 4, 8, 2],
    [6, 0, 4, 8, 2, 7, 1, 5, 9, 3],
    [7, 1, 5, 9, 3, 8, 2, 6, 0, 4],
    [8, 2, 6, 0, 4, 9, 3, 7, 1, 5],
    [9, 3, 7, 1, 5, 0, 4, 8, 2, 6],
]

L4 = [
    [0, 5, 0, 5, 0, 5, 0, 5, 0, 5], # This is not a Latin square!
]
# I cannot invent these. I will return the 2 known squares.

def _is_latin(sq):
    n = 10
    for i in range(n):
        if set(sq[i]) != set(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if set(col) != set(range(n)):
            return False
    return True

def _is_orthogonal(a, b):
    seen = set()
    for i in range(10):
        for j in range(10):
            p = (a[i][j], b[i][j])
            if p in seen:
                return False
            seen.add(p)
    return True

def _validate(squares):
    for sq in squares:
        if not _is_latin(sq):
            return False
    for i in range(len(squares)):
        for j in range(i + 1, len(squares)):
            if not _is_orthogonal(squares[i], squares[j]):
                return False
    return True

def construct():
    t0 = time.time()
    result = [L1, L2]
    if _validate(result):
        return result
    return [L1]
# EVOLVE-BLOCK-END
