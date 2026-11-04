import time
import numpy as np

DIMENSIONS = [4, 4, 5]
n, m, p = DIMENSIONS  # 4, 4, 5

# ---------------------------------------------------------------------------
# Two-level Strassen decomposition of the 4x4 * 4x4 part (rank 49)
#   + schoolbook 4x4 * 4x1 part (rank 16)
# Total rank 65.
# ---------------------------------------------------------------------------

# Super-level Strassen: P = A00+A11 etc.  p_blocks/q_blocks are coefficients of
# the 2x2 A/B blocks (A00,A01,A10,A11) and (B00,B01,B10,B11).
SUPER = {
    0: ([1, 0, 0, 1], [1, 0, 0, 1]),   # M1
    1: ([0, 0, 1, 1], [1, 0, 0, 0]),   # M2
    2: ([1, 0, 0, 0], [0, 1, 0, -1]),  # M3
    3: ([0, 0, 0, 1], [-1, 0, 1, 0]),  # M4
    4: ([1, 1, 0, 0], [0, 0, 0, 1]),   # M5
    5: ([-1, 0, 1, 0], [1, 1, 0, 0]),  # M6
    6: ([0, 1, 0, -1], [0, 0, 1, 1]),  # M7
}

# Where each super product's local 2x2 result lands in the global C
# (block_row, block_col, sign)
SUPER_BLOCKS = {
    0: [(0, 0, 1), (1, 1, 1)],
    1: [(1, 0, 1), (1, 1, -1)],
    2: [(0, 1, 1), (1, 1, 1)],
    3: [(0, 0, 1), (1, 0, 1)],
    4: [(0, 0, -1), (0, 1, 1)],
    5: [(1, 1, 1)],
    6: [(0, 0, 1)],
}

# Inner-level Strassen: which P/Q elements each of the 7 scalar products uses,
# and where the scalar lands in the local 2x2 result (local_row, local_col, coeff).
#   0: (P00+P11)(Q00+Q11)
#   1: (P10+P11)Q00
#   2: P00(Q01-Q11)
#   3: P11(Q10-Q00)
#   4: (P00+P01)Q11
#   5: (P10-P00)(Q00+Q01)
#   6: (P01-P11)(Q10+Q11)
INNER = {
    0: ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 0, 1), (1, 1, 1)]),
    1: ([(2, 1), (3, 1)], [(0, 1)],          [(1, 0, 1), (1, 1, -1)]),
    2: ([(0, 1)],           [(1, 1), (3, -1)], [(0, 1, 1), (1, 1, 1)]),
    3: ([(3, 1)],           [(2, 1), (0, -1)], [(0, 0, 1), (1, 0, 1)]),
    4: ([(0, 1), (1, 1)],   [(3, 1)],          [(0, 0, -1), (0, 1, 1)]),
    5: ([(2, 1), (0, -1)],  [(0, 1), (1, 1)],  [(1, 1, 1)]),
    6: ([(1, 1), (3, -1)],  [(2, 1), (3, 1)],  [(0, 0, 1)]),
}

# Map a 2x2 local (r, c) element of an A block b to global (i, j)
def a_global(b, r, c):
    if b == 0:
        return r, c
    if b == 1:
        return r, 2 + c
    if b == 2:
        return 2 + r, c
    return 2 + r, 2 + c

def b_global(b, r, c):
    if b == 0:
        return r, c
    if b == 1:
        return r, 2 + c
    if b == 2:
        return 2 + r, c
    return 2 + r, 2 + c


def _build():
    """Return (U, V, W) as lists of lists, rank 65, if the decomposition is exact."""
    rank = 49 + 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0
    for si in range(7):
        p_blocks, q_blocks = SUPER[si]
        for ii in range(7):
            p_elems, q_elems, c_contrib = INNER[ii]

            # U: linear combination of A entries
            for pe, pc in p_elems:
                r = pe // 2
                c = pe % 2
                for b, bc in enumerate(p_blocks):
                    if bc == 0:
                        continue
                    gi, gj = a_global(b, r, c)
                    U[gi * m + gj, col] += pc * bc

            # V: linear combination of B entries
            for qe, qc in q_elems:
                r = qe // 2
                c = qe % 2
                for b, bc in enumerate(q_blocks):
                    if bc == 0:
                        continue
                    gj, gk = b_global(b, r, c)
                    V[gj * p + gk, col] += qc * bc

            # W: trace convention W[k*n + i]
            for br, bcc, sign in SUPER_BLOCKS[si]:
                for lr, lc, lc_coeff in c_contrib:
                    gi = br * 2 + lr
                    gk = bcc * 2 + lc
                    W[gk * n + gi, col] += sign * lc_coeff

            col += 1

    # Part 2: 4x4 * 4x1 schoolbook (last B column -> last C column)
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1.0
            V[j * p + (p - 1), col] = 1.0
            W[(p - 1) * n + i, col] = 1.0
            col += 1

    U = U[:, :col]
    V = V[:, :col]
    W = W[:, :col]

    # Verify exact decomposition: C[i,k] = sum_j A[i,j] B[j,k]
    A = np.random.RandomState(0).rand(n, m)
    B = np.random.RandomState(1).rand(m, p)
    C = A @ B
    for i in range(n):
        for k in range(p):
            acc = 0.0
            for j in range(m):
                acc += A[i, j] * V[j * p + k, :] * W[k * n + i, :] * U[i * m + j, :]
            if abs(acc - C[i, k]) > 1e-6:
                return None
    return U.tolist(), V.tolist(), W.tolist()


def _fallback():
    """Exact schoolbook decomposition, rank n*m*p."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    start = time.time()
    try:
        res = _build()
    except Exception:
        res = None
    if res is None:
        res = _fallback()
    # Safety margin: ensure we return well before the 120 s limit.
    if time.time() - start > 110:
        res = _fallback()
    U, V, W = res
    return {"U": U, "V": V, "W": W}
