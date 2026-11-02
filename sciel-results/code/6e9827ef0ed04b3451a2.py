"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import random
from fractions import Fraction
import numpy as np

def _build_target(n, m, p):
    """Return the matrix-multiplication tensor as a dict of 1s (sparse)."""
    tgt = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tgt[(i * m + j, j * p + k, k * n + i)] = 1
    return tgt

def _schoolbook(n, m, p):
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
    return U, V, W

def _verify(U, V, W, tgt, n, m, p):
    """Exact check that sum_r U[a,r]V[b,r]W[c,r] reproduces the target."""
    R = len(U[0])
    # Represent each factor column as a dict of nonzero entries.
    def col_dict(mat, r):
        d = {}
        for a in range(len(mat)):
            v = mat[a][r]
            if v != 0:
                d[a] = v
        return d
    # Precompute
    cols = []
    for r in range(R):
        cols.append((col_dict(U, r), col_dict(V, r), col_dict(W, r)))
    acc = {}
    for cu, cv, cw in cols:
        for a, ua in cu.items():
            for b, vb in cv.items():
                uv = ua * vb
                for c, wc in cw.items():
                    acc[(a, b, c)] = acc.get((a, b, c), 0) + uv * wc
    # Every target entry must equal 1
    for key, val in tgt.items():
        if acc.get(key, 0) != val:
            return False
    # No spurious entries
    for key, val in acc.items():
        if val != 0 and key not in tgt:
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115  # Keep safety margin before 120s limit

    # Start with the schoolbook decomposition (rank = 72)
    U, V, W = _schoolbook(n, m, p)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
