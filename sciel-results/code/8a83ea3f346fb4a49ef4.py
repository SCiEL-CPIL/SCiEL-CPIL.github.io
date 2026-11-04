"""
Matrix multiplication 2x4x7 via tensor decomposition.
Constructs a rank-50 exact decomposition:
  - 2x2x4 (rank 14): two 2x2x2 Strassen blocks
  - 2x2x2 (rank 7):  one 2x2x2 Strassen block
  - 2x2x2 (rank 7):  one 2x2x2 Strassen block
  - 2x4x1 (rank 8):  rank-1 terms
Total: 14 + 7 + 7 + 8 = 36
Wait: 14+7+7+8 = 36? No, 14+7=21, +7=28, +8=36. That's 36.
Actually the current program uses rank 50. Let me check: 28+14+8=50.
The 28 is 4 blocks of Strassen (7 each), 14 is 2 blocks of Strassen, 8 is rank-1.
So 4+2=6 Strassen blocks = 42, plus 8 = 50.
But 2x4x4 needs 2x(2x2x4) = 2*14 = 28. Yes 4 blocks.
2x4x2 needs 2x(2x2x2) = 2*7 = 14. Yes 2 blocks.
2x4x1 needs 8. Total 50.
"""
import time
_DIMENSIONS = [2, 4, 7]
_START_TIME = time.time()
_TIME_LIMIT = 110  # safety margin before 120s evaluator kill

def construct():
    n, m, p = _DIMENSIONS
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        # M2
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        # M3
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        # M4
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        # M5
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        # M6
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        # M7
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # Part 1: 2x4x4 (k=0..3) - 4 blocks of 2x2x2
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)

    # Part 2: 2x4x2 (k=4..5) - 2 blocks of 2x2x2
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)

    # Part 3: 2x4x1 (k=6) - 8 rank-1 terms
    for i in range(n):
        for j in range(m):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1

    assert col == rank, f"Rank mismatch: used {col}, expected {rank}"
    if time.time() - _START_TIME > _TIME_LIMIT:
        pass  # still return, construction is O(1)
    return {"U": U, "V": V, "W": W}
