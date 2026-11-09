"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
TIME_LIMIT = 130.0  # Safety margin before 150s kill


def _schoolbook(n, m, p):
    """Build the trivial schoolbook decomposition with rank n*m*p."""
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
    return U, V, W, rank


def _strassen_2x2x2():
    """
    Return the 7-term Strassen decomposition for 2x2x2 matrix multiplication.
    U: 4x7, V: 4x7, W: 4x7
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # The 7 Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    U = [
        [1, 0, 1, 0, 1, 0, 0],  # A11
        [0, 1, 0, 0, 0, -1, 0], # A21
        [0, 0, 0, 0, 0, 0, -1], # A22
        [0, 0, 1, 0, 1, 0, -1], # A12
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],  # B11
        [0, 0, 1, 0, 0, 0, 1],   # B21
        [1, 1, -1, -1, 1, 1, 1], # B12
        [0, 0, 0, 0, 1, 0, 0],   # B22
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],  # C11 (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],   # C12 (k=1,i=0)
        [0, 1, 0, 1, 0, 0, 0],   # C21 (k=0,i=1)
        [1, -1, 1, 0, 0, 1, 0],  # C22 (k=1,i=1)
    ]
    return U, V, W, 7


def _combine_along_m(n, m1, m2, p, decomp1, decomp2):
    """
    Combine two decompositions for (n, m1, p) and (n, m2, p) into (n, m1+m2, p).
    decomp1 = (U1, V1, W1, r1), decomp2 = (U2, V2, W2, r2)
    """
    U1, V1, W1, r1 = decomp1
    U2, V2, W2, r2 = decomp2
    m = m1 + m2
    rank = r1 + r2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # U: first m1 rows use U1, next m2 rows use U2 (offset by r1)
    for i in range(n):
        for j in range(m1):
            for c in range(r1):
                U[i * m + j][c] = U1[i * m1 + j][c]
        for j in range(m2):
            for c in range(r2):
                U[i * m + m1 + j][r1 + c] = U2[i * m2 + j][c]

    # V: first m1 rows use V1, next m2 rows use V2 (offset by r1)
    for j in range(m1):
        for k in range(p):
            for c in range(r1):
                V[j * p + k][c] = V1[j * p + k][c]
    for j in range(m2):
        for k in range(p):
            for c in range(r2):
                V[(m1 + j) * p + k][r1 + c] = V2[j * p + k][c]

    # W: both contribute to same C, so sum
    for i in range(n):
        for k in range(p):
            for c in range(r1):
                W[k * n + i][c] += W1[k * n + i][c]
            for c in range(r2):
                W[k * n + i][r1 + c] += W2[k * n + i][c]

    return U, V, W, rank


def _combine_along_n(n1, n2, m, p, decomp1, decomp2):
    """
    Combine two decompositions for (n1, m, p) and (n2, m, p) into (n1+n2, m, p).
    """
    U1, V1, W1, r1 = decomp1
    U2, V2, W2, r2 = decomp2
    n = n1 + n2
    rank = r1 + r2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # U: first n1 rows use U1, next n2 rows use U2
    for i in range(n1):
        for j in range(m):
            for c in range(r1):
                U[i * m + j][c] = U1[i * m + j][c]
    for i in range(n2):
        for j in range(m):
            for c in range(r2):
                U[(n1 + i) * m + j][r1 + c] = U2[i * m + j][c]

    # V: both use same B, so sum
    for j in range(m):
        for k in range(p):
            for c in range(r1):
                V[j * p + k][c] += V1[j * p + k][c]
            for c in range(r2):
                V[j * p + k][r1 + c] += V2[j * p + k][c]

    # W: first n1 rows use W1, next n2 rows use W2
    for i in range(n1):
        for k in range(p):
            for c in range(r1):
                W[k * n + i][c] = W1[k * n1 + i][c]
    for i in range(n2):
        for k in range(p):
            for c in range(r2):
                W[k * n + n1 + i][r1 + c] = W2[k * n2 + i][c]

    return U, V, W, rank


def _combine_along_p(n, m, p1, p2, decomp1, decomp2):
    """
    Combine two decompositions for (n, m, p1) and (n, m, p2) into (n, m, p1+p2).
    """
    U1, V1, W1, r1 = decomp1
    U2, V2, W2, r2 = decomp2
    p = p1 + p2
    rank = r1 + r2
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # U: both use same A, so sum
    for i in range(n):
        for j in range(m):
            for c in range(r1):
                U[i * m + j][c] += U1[i * m + j][c]
            for c in range(r2):
                U[i * m + j][r1 + c] += U2[i * m + j][c]

    # V: first p1 cols use V1, next p2 cols use V2
    for j in range(m):
        for k in range(p1):
            for c in range(r1):
                V[j * p + k][c] = V1[j * p1 + k][c]
        for k in range(p2):
            for c in range(r2):
                V[j * p + p1 + k][r1 + c] = V2[j * p2 + k][c]

    # W: first p1 cols use W1, next p2 cols use W2
    for i in range(n):
        for k in range(p1):
            for c in range(r1):
                W[k * n + i][c] = W1[k * n + i][c]
        for k in range(p2):
            for c in range(r2):
                W[(p1 + k) * n + i][r1 + c] = W2[k * n2 + i][c]

    return U, V, W, rank


def _build_decomp(n, m, p):
    """
    Build a decomposition for M_{n,m,p} using recursive splitting with Strassen base case.
    Returns (U, V, W, rank).
    """
    if time.time() - _start_time > TIME_LIMIT:
        # Return schoolbook as fallback
        return _schoolbook(n, m, p)

    # Base cases
    if n == 1 and m == 1 and p == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W, 1

    if n == 1:
        rank = m * p
        U = [[0] * rank for _ in range(m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p)]
        col = 0
        for j in range(m):
            for k in range(p):
                U[j][col] = 1
                V[j * p + k][col] = 1
                W[k][col] = 1
                col += 1
        return U, V, W, rank

    if m == 1:
        rank = n * p
        U = [[0] * rank for _ in range(n)]
        V = [[0] * rank for _ in range(p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i][col] = 1
                V[k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        return U, V, W, rank

    if p == 1:
        rank = n * m
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m)]
        W = [[0] * rank for _ in range(n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i * m + j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W, rank

    # Strassen base case for 2x2x2
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2x2()

    # Try all possible splits and pick the best one
    best_rank = n * m * p
    best_decomp = None

    # Try splitting m: (n, m1, p) + (n, m2, p)
    for m1 in range(1, m):
        m2 = m - m1
        d1 = _build_decomp(n, m1, p)
        d2 = _build_decomp(n, m2, p)
        r1, r2 = d1[3], d2[3]
        candidate_rank = r1 + r2
        if candidate_rank < best_rank:
            best_rank = candidate_rank
            combined = _combine_along_m(n, m1, m2, p, d1, d2)
            best_decomp = combined

    # Try splitting n: (n1, m, p) + (n2, m, p)
    for n1 in range(1, n):
        n2 = n - n1
        d1 = _build_decomp(n1, m, p)
        d2 = _build_decomp(n2, m, p)
        r1, r2 = d1[3], d2[3]
        candidate_rank = r1 + r2
        if candidate_rank < best_rank:
            best_rank = candidate_rank
            combined = _combine_along_n(n1, n2, m, p, d1, d2)
            best_decomp = combined

    # Try splitting p: (n, m, p1) + (n, m, p2)
    for p1 in range(1, p):
        p2 = p - p1
        d1 = _build_decomp(n, m, p1)
        d2 = _build_decomp(n, m, p2)
        r1, r2 = d1[3], d2[3]
        candidate_rank = r1 + r2
        if candidate_rank < best_rank:
            best_rank = candidate_rank
            combined = _combine_along_p(n, m, p1, p2, d1, d2)
            best_decomp = combined

    if best_decomp is None:
        return _schoolbook(n, m, p)

    return best_decomp


def construct():
    n, m, p = DIMENSIONS
    U, V, W, rank = _build_decomp(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
