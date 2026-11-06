"""
Matrix multiplication 6x6x6 tensor decomposition.

Strategy:
  Partition the 6x6 matrices into a 3x3 grid of 2x2 blocks and apply the
  classic rank-7 Strassen algorithm to every block position, giving a total
  rank of 3*3*3*7 = 189.  This beats the schoolbook rank 216 by a wide margin.

  The construction is exact (integer coefficients only), so the returned
  factor matrices reproduce the 6x6 matrix multiplication tensor with
  no approximation.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]
_DEADLINE = 120.0
_START_TIME = None


def _now_ok(deadline=120.0, margin=1.0):
    """Cheap wall-clock guard that returns True while we are safely under deadline."""
    return (time.time() - _START_TIME) < (deadline - margin)


def _build_strassen_grid():
    """Build a 3x3 grid of 2x2 Strassen blocks (rank 189).

    Row conventions:
        U[i*m + j, r] : coefficient of A[i, j] in product r
        V[j*p + k, r] : coefficient of B[j, k] in product r
        W[k*n + i, r] : coefficient of C[i, k] in product r   (official trace convention)

    For each triple of 2x2 block positions (i_b, k_b, j_b) with i_b, k_b, j_b in {0,1,2},
    the 7 Strassen products mix the 4 sub-blocks of A[i_b, k_b], B[k_b, j_b]
    and contribute to the 4 sub-blocks of C[i_b, j_b].
    """
    n, m, p = DIMENSIONS
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for i_b in range(3):
        for k_b in range(3):
            for j_b in range(3):
                # Row helpers (a, b) in {0,1} indexes the 2x2 sub-block within the 2x2 block
                def u_row(a, b):
                    return (2 * i_b + a) * m + (2 * k_b + b)

                def v_row(c, d):
                    return (2 * k_b + c) * p + (2 * j_b + d)

                def w_row(e, f):
                    return (2 * j_b + f) * n + (2 * i_b + e)

                # M1 = (A11 + A22) * (B11 + B22)
                U[u_row(0, 0)][col] = 1; U[u_row(1, 1)][col] = 1
                V[v_row(0, 0)][col] = 1; V[v_row(1, 1)][col] = 1
                W[w_row(0, 0)][col] = 1; W[w_row(1, 1)][col] = 1
                col += 1

                # M2 = (A21 + A22) * B11
                U[u_row(1, 0)][col] = 1; U[u_row(1, 1)][col] = 1
                V[v_row(0, 0)][col] = 1
                W[w_row(1, 0)][col] = 1; W[w_row(1, 1)][col] = -1
                col += 1

                # M3 = A11 * (B12 - B22)
                U[u_row(0, 0)][col] = 1
                V[v_row(0, 1)][col] = 1; V[v_row(1, 1)][col] = -1
                W[w_row(0, 1)][col] = 1; W[w_row(1, 1)][col] = 1
                col += 1

                # M4 = A22 * (B21 - B11)
                U[u_row(1, 1)][col] = 1
                V[v_row(1, 0)][col] = 1; V[v_row(0, 0)][col] = -1
                W[w_row(0, 0)][col] = 1; W[w_row(1, 0)][col] = 1
                col += 1

                # M5 = (A11 + A12) * B22
                U[u_row(0, 0)][col] = 1; U[u_row(0, 1)][col] = 1
                V[v_row(1, 1)][col] = 1
                W[w_row(0, 0)][col] = -1; W[w_row(0, 1)][col] = 1
                col += 1

                # M6 = (A21 - A11) * (B11 + B12)
                U[u_row(1, 0)][col] = 1; U[u_row(0, 0)][col] = -1
                V[v_row(0, 0)][col] = 1; V[v_row(0, 1)][col] = 1
                W[w_row(1, 1)][col] = 1
                col += 1

                # M7 = (A12 - A22) * (B21 + B22)
                U[u_row(0, 1)][col] = 1; U[u_row(1, 1)][col] = -1
                V[v_row(1, 0)][col] = 1; V[v_row(1, 1)][col] = 1
                W[w_row(0, 0)][col] = 1
                col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition: U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r
    should equal 1 for every (i, j, k) triple."""
    Ua = np.asarray(U, dtype=np.float64)
    Va = np.asarray(V, dtype=np.float64)
    Wa = np.asarray(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.dot(Ua[i * m + j], Va[j * p + k] * Wa[k * n + i]))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _build_schoolbook():
    """Ultimate fallback: the trivial schoolbook decomposition of rank 216."""
    n, m, p = DIMENSIONS
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


def construct():
    """Return {'U', 'V', 'W'} factor matrices for an exact rank-189 decomposition
    of the 6x6x6 matrix multiplication tensor (3x3 grid of 2x2 Strassen blocks)."""
    global _START_TIME
    _START_TIME = time.time()

    n, m, p = DIMENSIONS

    U, V, W = _build_strassen_grid()

    # Verify before returning; fall back to schoolbook if anything is off.
    if _now_ok():
        ok = _verify(U, V, W, n, m, p)
    else:
        ok = False

    if not ok:
        U, V, W = _build_schoolbook()

    return {"U": U, "V": V, "W": W}
