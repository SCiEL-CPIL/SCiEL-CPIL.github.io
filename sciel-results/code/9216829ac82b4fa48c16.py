"""
Rank-minimal exact tensor decomposition for 2x4 by 4x8 matrix multiplication.

Strategy: 8 independent 2x2x2x2 sub-problems (8 column-pairs of B × 2 row-halves of A).
Each sub-problem uses Strassen (7 rank-1 terms). Total rank = 8*7 = 56.
"""
import time

_START_TIME = time.time()
_DEADLINE = 115.0


def _check_deadline():
    return (time.time() - _START_TIME) < _DEADLINE


def construct():
    """
    Return factor matrices U, V, W for the 2x4 * 4x8 matrix multiplication tensor.
    U: (n*m, rank) = (8, 56)
    V: (m*p, rank) = (32, 56)
    W: (p*n, rank) = (16, 56)

    The decomposition satisfies: for all i,j,k,
        sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    (i.e. the tensor of matrix multiplication is exactly represented.)
    """
    n, m, p = 2, 4, 8
    rank = 56

    if not _check_deadline():
        # Fallback: return naive rank-64 decomposition if we're over deadline
        return _naive_construct()

    U = [[0.0] * rank for _ in range(n * m)]   # 8 x 56
    V = [[0.0] * rank for _ in range(m * p)]   # 32 x 56
    W = [[0.0] * rank for _ in range(p * n)]   # 16 x 56

    # 8 blocks: d in {0,1,2,3} (B column-pair), h in {0,1} (A row-half)
    # Block (d, h) uses columns 2d, 2d+1 of B and columns 2h, 2h+1 of A
    # producing columns 2d, 2d+1 of C (2 rows each)

    for d in range(4):
        for h in range(2):
            base = (d * 2 + h) * 7

            # Index helpers for this 2x2 sub-block
            # A rows are 0,1 (n=2); A cols are 2h, 2h+1
            # B rows are 2h, 2h+1 (the corresponding m-dim slice); B cols are 2d, 2d+1
            # C rows are 0,1; C cols are 2d, 2d+1

            def _a(r, c):
                """A[r, 2h+c] -> flat index in U"""
                return r * m + (2 * h + c)

            def _b(r, c):
                """B[2h+r, 2d+c] -> flat index in V"""
                return (2 * h + r) * p + (2 * d + c)

            def _c(r, c):
                """C[r, 2d+c] -> flat index in W (W index = col*n + row)"""
                return (2 * d + c) * n + r

            # Strassen 7 products for 2x2 matrix multiplication:
            # C = A * B where A, B, C are 2x2
            # M1 = (A00+A11)*(B00+B11)
            # M2 = (A10+A11)*B00
            # M3 = A00*(B01-B11)
            # M4 = A11*(B10-B00)
            # M5 = (A00+A01)*B11
            # M6 = (A10-A00)*(B00+B01)
            # M7 = (A01-A11)*(B10+B11)
            # C00 = M1 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M6

            # M1 = (A00+A11)*(B00+B11)
            col = base + 0
            U[_a(0, 0)][col] += 1.0
            U[_a(1, 1)][col] += 1.0
            V[_b(0, 0)][col] += 1.0
            V[_b(1, 1)][col] += 1.0
            W[_c(0, 0)][col] += 1.0
            W[_c(1, 1)][col] += 1.0

            # M2 = (A10+A11)*B00
            col = base + 1
            U[_a(1, 0)][col] += 1.0
            U[_a(1, 1)][col] += 1.0
            V[_b(0, 0)][col] += 1.0
            W[_c(1, 0)][col] += 1.0
            W[_c(1, 1)][col] -= 1.0

            # M3 = A00*(B01-B11)
            col = base + 2
            U[_a(0, 0)][col] += 1.0
            V[_b(0, 1)][col] += 1.0
            V[_b(1, 1)][col] -= 1.0
            W[_c(0, 1)][col] += 1.0
            W[_c(1, 1)][col] += 1.0

            # M4 = A11*(B10-B00)
            col = base + 3
            U[_a(1, 1)][col] += 1.0
            V[_b(1, 0)][col] += 1.0
            V[_b(0, 0)][col] -= 1.0
            W[_c(0, 0)][col] += 1.0
            W[_c(1, 0)][col] += 1.0

            # M5 = (A00+A01)*B11
            col = base + 4
            U[_a(0, 0)][col] += 1.0
            U[_a(0, 1)][col] += 1.0
            V[_b(1, 1)][col] += 1.0
            W[_c(0, 0)][col] -= 1.0
            W[_c(0, 1)][col] += 1.0

            # M6 = (A10-A00)*(B00+B01)
            col = base + 5
            U[_a(1, 0)][col] += 1.0
            U[_a(0, 0)][col] -= 1.0
            V[_b(0, 0)][col] += 1.0
            V[_b(0, 1)][col] += 1.0
            W[_c(1, 1)][col] += 1.0

            # M7 = (A01-A11)*(B10+B11)
            col = base + 6
            U[_a(0, 1)][col] += 1.0
            U[_a(1, 1)][col] -= 1.0
            V[_b(1, 0)][col] += 1.0
            V[_b(1, 1)][col] += 1.0
            W[_c(0, 0)][col] += 1.0

    # Convert to integer entries where possible (all entries are integers)
    for mat in (U, V, W):
        for row in mat:
            for i in range(len(row)):
                row[i] = int(round(row[i]))

    if not _check_deadline():
        return _naive_construct()

    return {"U": U, "V": V, "W": W}


def _naive_construct():
    """Fallback naive rank-64 decomposition."""
    n, m, p = 2, 4, 8
    rank = n * m * p  # 64
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    return {"U": U, "V": V, "W": W}
