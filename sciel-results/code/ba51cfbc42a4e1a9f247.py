"""Tensor decomposition for 4x5x5 matrix multiplication.

Strategy: recursive Strassen-like block partitioning.
- Split all three dimensions (m, n, p) at every possible cut.
- Use the rank-7 Strassen decomposition for the 2x2x2 base case.
- Use schoolbook (rank m*n*p) for small/degenerate sub-problems.
- Cache sub-problems to avoid recomputation.
- Verify the final decomposition with the trace convention W[k*n+i, r].
"""
import time
import numpy as np
from functools import lru_cache

DIMENSIONS = [4, 5, 5]
_START_TIME = time.time()
TIME_LIMIT = 135  # safety margin before the 150 s kill


def _timeout():
    if time.time() - _START_TIME > TIME_LIMIT:
        raise TimeoutError()


def _schoolbook(m, n, p):
    """Schoolbook decomposition with rank m*n*p (as numpy arrays)."""
    rank = m * n * p
    U = np.zeros((m * n, rank), dtype=float)
    V = np.zeros((n * p, rank), dtype=float)
    W = np.zeros((p * m, rank), dtype=float)
    col = 0
    for i in range(m):
        for j in range(n):
            for k in range(p):
                U[i * n + j, col] = 1
                V[j * p + k, col] = 1
                W[k * m + i, col] = 1
                col += 1
    return U, V, W


def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2 (numpy arrays).

    U: (4,7) indexed [i*2+j, c]; V: (4,7) indexed [j*2+k, c];
    W: (4,7) indexed [k*2+i, c].
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1   # C00 += M1
    W[3, 0] = 1   # C11 += M1

    # M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1   # C10 += M2

    # M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1   # C01 += M3
    W[3, 2] = -1  # C11 -= M3

    # M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = -1  # C00 -= M4
    W[2, 3] = 1   # C10 += M4

    # M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1   # C01 += M5

    # M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[0, 5] = 1   # C00 += M6

    # M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1   # C11 += M7

    return U, V, W


def _rank_of(decomp):
    U = decomp[0]
    return U.shape[1] if isinstance(U, np.ndarray) else len(U[0])


@lru_cache(maxsize=None)
def _decompose(m, n, p):
    """Return (U, V, W) numpy arrays for m x n * n x p multiplication.

    U: (m*n, r), V: (n*p, r), W: (p*m, r) with W[k*m+i, c] convention.
    """
    _timeout()

    if m == 0 or n == 0 or p == 0:
        return (np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0)))

    # Strassen base case
    if m == 2 and n == 2 and p == 2:
        return _strassen_2x2()

    # Degenerate / small cases: schoolbook
    if m == 1 or n == 1 or p == 1:
        return _schoolbook(m, n, p)

    # Recursively try every 3-way split; keep the minimum-rank combination.
    best = None
    best_rank = m * n * p  # schoolbook upper bound

    for m1 in range(1, m):
        m2 = m - m1
        for n1 in range(1, n):
            n2 = n - n1
            for p1 in range(1, p):
                p2 = p - p1
                _timeout()
                try:
                    d1 = _decompose(m1, n1, p1)  # A11 * B11
                    d2 = _decompose(m1, n1, p2)  # A11 * B12
                    d3 = _decompose(m1, n2, p1)  # A12 * B21
                    d4 = _decompose(m1, n2, p2)  # A12 * B22
                    d5 = _decompose(m2, n1, p1)  # A21 * B11
                    d6 = _decompose(m2, n1, p2)  # A21 * B12
                    d7 = _decompose(m2, n2, p1)  # A22 * B21
                    d8 = _decompose(m2, n2, p2)  # A22 * B22

                    total_rank = sum(_rank_of(d) for d in
                                     (d1, d2, d3, d4, d5, d6, d7, d8))
                    if total_rank < best_rank:
                        best_rank = total_rank
                        best = _combine(d1, d2, d3, d4, d5, d6, d7, d8,
                                        m, n, p, m1, n1, p1)
                except TimeoutError:
                    raise
                except Exception:
                    continue

    if best is not None:
        return best
    return _schoolbook(m, n, p)


def _combine(d1, d2, d3, d4, d5, d6, d7, d8,
             m, n, p, m1, n1, p1):
    """Combine the 8 block sub-decompositions into one for the full m x n * n x p.

    Block layout (A is m x n, B is n x p, C is m x p):
      d1: rows[0:m1], Acols[0:n1], Bcols[0:p1]
      d2: rows[0:m1], Acols[0:n1], Bcols[p1:p]
      d3: rows[0:m1], Acols[n1:n], Bcols[0:p1]
      d4: rows[0:m1], Acols[n1:n], Bcols[p1:p]
      d5: rows[m1:m], Acols[0:n1], Bcols[0:p1]
      d6: rows[m1:m], Acols[0:n1], Bcols[p1:p]
      d7: rows[m1:m], Acols[n1:n], Bcols[0:p1]
      d8: rows[m1:m], Acols[n1:n], Bcols[p1:p]
    """
    m2 = m - m1
    n2 = n - n1
    p2 = p - p1
    total_rank = sum(_rank_of(d) for d in
                     (d1, d2, d3, d4, d5, d6, d7, d8))

    U = np.zeros((m * n, total_rank), dtype=float)
    V = np.zeros((n * p, total_rank), dtype=float)
    W = np.zeros((p * m, total_rank), dtype=float)

    col = 0

    def add(U_s, V_s, W_s, mi, ni, pi, off_m, off_n, off_p):
        nonlocal col
        r = _rank_of((U_s, V_s, W_s))
        for c in range(r):
            for i in range(mi):
                for j in range(ni):
                    U[(off_m + i) * n + (off_n + j), col + c] = U_s[i * ni + j, c]
            for j in range(ni):
                for k in range(pi):
                    V[(off_n + j) * p + (off_p + k), col + c] = V_s[j * pi + k, c]
            for k in range(pi):
                for i in range(mi):
                    W[(off_p + k) * m + (off_m + i), col + c] = W_s[k * mi + i, c]
        col += r

    add(d1[0], d1[1], d1[2], m1, n1, p1, 0,    0,    0)
    add(d2[0], d2[1], d2[2], m1, n1, p2, 0,    0,    p1)
    add(d3[0], d3[1], d3[2], m1, n2, p1, 0,    n1,   0)
    add(d4[0], d4[1], d4[2], m1, n2, p2, 0,    n1,   p1)
    add(d5[0], d5[1], d5[2], m2, n1, p1, m1,   0,    0)
    add(d6[0], d6[1], d6[2], m2, n1, p2, m1,   0,    p1)
    add(d7[0], d7[1], d7[2], m2, n2, p1, m1,   n1,   0)
    add(d8[0], d8[1], d8[2], m2, n2, p2, m1,   n1,   p1)

    assert col == total_rank, f"column offset {col} != total rank {total_rank}"
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p):
    """Verify U,V,W decompose the matrix-multiplication tensor (trace convention)."""
    rank = U.shape[1]
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != rank or V.shape[1] != rank or W.shape[1] != rank:
        return False

    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B

    C_comp = np.zeros((n, p))
    for c in range(rank):
        U_c = U[:n * m, c].reshape(n, m)
        V_c = V[:m * p, c].reshape(m, p)
        W_c = W[:p * n, c].reshape(p, n)
        u_a = np.sum(U_c * A, axis=1)      # (n,)
        b_v = np.sum(B * V_c, axis=0)      # (p,)
        C_comp += W_c.T * np.outer(u_a, b_v)

    return bool(np.allclose(C_comp, C_true, atol=1e-8))


def construct():
    global _START_TIME
    _START_TIME = time.time()
    deadline = _START_TIME + 138  # safety margin before 150 s kill

    n, m, p = DIMENSIONS  # 4, 5, 5

    # Schoolbook fallback (always valid).
    U_fb, V_fb, W_fb = _schoolbook(n, m, p)
    best_result = {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    best_rank = n * m * p

    if time.time() < deadline:
        try:
            U_blk, V_blk, W_blk = _decompose(n, m, p)
            if _verify_decomposition(U_blk, V_blk, W_blk, n, m, p):
                rank_blk = U_blk.shape[1]
                if rank_blk < best_rank:
                    best_rank = rank_blk
                    best_result = {"U": U_blk.tolist(),
                                   "V": V_blk.tolist(),
                                   "W": W_blk.tolist()}
        except Exception:
            pass

    if time.time() > deadline:
        return best_result

    return best_result
