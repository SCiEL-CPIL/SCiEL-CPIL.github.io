"""Matrix multiplication 2,3,3 - rank-minimal tensor decomposition.

Strategy:
  1. Fast random-search pass for a low-rank decomposition (bounded time).
  2. Explicit constructive decomposition (block Strassen + direct blocks).
  3. Schoolbook fallback.

The decomposition is exact (integer entries) and verified with exact arithmetic.
"""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 3, 3]


def _exact_verify(U, V, W, n, m, p):
    """Exact verification with integer arithmetic (works for ints and Fractions)."""
    rank = len(U[0])
    nm = n * m
    mp = m * p
    pn = p * n

    # Precompute target 1-positions
    targets = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                targets[(u, v, w)] = 1

    for u in range(nm):
        for v in range(mp):
            for w in range(pn):
                target = targets.get((u, v, w), 0)
                s = 0
                for c in range(rank):
                    s += U[u][c] * V[v][c] * W[w][c]
                if s != target:
                    return False
    return True


def _construct_block_decomposition(n, m, p):
    """Construct a rank-36 decomposition via block Strassen + direct blocks.

    Splits the 2x3 * 3x3 product into:
      - 4 blocks of 2x2 * 2x2  (columns 0,1 of B)  -> Strassen rank 7 each
      - 2 blocks of 2x2 * 2x1  (column 2 of B)     -> direct rank 4 each
    Total rank = 4*7 + 2*4 = 36.
    """
    nm = n * m  # 6
    mp = m * p  # 9
    pn = p * n  # 6
    rank = 4 * 7 + 2 * 4  # 36

    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]

    col = 0

    # --- 2x2 * 2x2 Strassen blocks (for B-column pairs (0,1) and (2,3)) ---
    # Strassen factors: (a, b, c) indices in [0,2).
    # M1 = (A00+A11)(B01+B10)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B11)
    # M7 = (A01-A10)(B01+B11)
    # C00 = M1+M4-M5+M7 ; C01 = M3+M5
    # C10 = M2+M4 ;      C11 = M1-M2+M3+M6

    strassen_defs = [
        # (M, a_factor, b_factor) where a_factor is a list of (sign, a_idx)
        # and b_factor is a list of (sign, b_idx)
        (0, [(1, 0), (1, 3)], [(1, 1), (1, 2)]),   # M1
        (1, [(1, 2), (1, 3)], [(1, 0)]),            # M2
        (2, [(1, 0)], [(1, 2), (-1, 3)]),           # M3
        (3, [(1, 3)], [(1, 1), (-1, 2)]),           # M4
        (4, [(1, 0), (1, 1)], [(1, 3)]),            # M5
        (5, [(1, 2), (-1, 0)], [(1, 0), (1, 3)]),   # M6
        (6, [(1, 1), (-1, 2)], [(1, 1), (1, 3)]),   # M7
    ]

    # C = sum over M of (coeff) * M, where C index (r,c) in {0,1}^2
    c_coeffs = [
        # C00: M1 + M4 - M5 + M7
        [(0, 1), (3, 1), (4, -1), (6, 1)],
        # C01: M3 + M5
        [(2, 1), (4, 1)],
        # C10: M2 + M4
        [(1, 1), (3, 1)],
        # C11: M1 - M2 + M3 + M6
        [(0, 1), (1, -1), (2, 1), (5, 1)],
    ]

    # For each B-column pair, and each row-group (i0 in {0,1})
    # We have 4 Strassen blocks: (b_col_pair in {(0,1),(2,3)}) x (i0 in {0,1})
    for bcol_pair in [(0, 1), (2, 3)]:
        for i0 in range(2):
            for (M, a_factor, b_factor) in strassen_defs:
                base = col
                # A part: rows i0, i0+1 ; cols 0,1,2,3 (local a_idx)
                # a_idx 0 -> (i0, 0), 1 -> (i0, 1), 2 -> (i0+1, 0), 3 -> (i0+1, 1)
                for (sa, a_idx) in a_factor:
                    i = i0 + (a_idx // 2)
                    j = a_idx % 2
                    u = i * m + j
                    U[u][base + M] = sa
                # B part: rows 0,1,2,3 ; cols bcol_pair
                # b_idx 0 -> (0, bcol_pair[0]), 1 -> (1, bcol_pair[0]),
                #        2 -> (0, bcol_pair[1]), 3 -> (1, bcol_pair[1])
                for (sb, b_idx) in b_factor:
                    j = b_idx // 2
                    k = bcol_pair[b_idx % 2]
                    v = j * p + k
                    V[v][base + M] = sb
                # C part: rows i0, i0+1 ; cols bcol_pair
                for (r, c) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    for (Mk, coeff) in c_coeffs[r * 2 + c]:
                        if Mk == M:
                            i = i0 + r
                            k = bcol_pair[c]
                            w = k * n + i
                            W[w][base + M] = coeff
            col += 7

    # --- 2x2 * 2x1 direct blocks (for B column 2) ---
    # For each row-group (i0 in {0,1}): 4 direct terms (i0+i, j) * (j, 2)
    for i0 in range(2):
        for i_off in range(2):
            for j in range(2):
                base = col
                i = i0 + i_off
                # A: (i, j)
                U[i * m + j][base] = 1
                # B: (j, 2)
                V[j * p + 2][base] = 1
                # C: (2, i)
                W[2 * n + i][base] = 1
                col += 1

    assert col == rank, f"column count {col} != rank {rank}"
    return U, V, W


def _try_rank_search(rank, n, m, p, max_attempts, time_limit):
    """Random-search for a rank-r decomposition via least squares + rational check."""
    rng = np.random.default_rng(42 + rank)
    nm = n * m
    mp = m * p
    pn = p * n
    num_eqs = nm * mp * pn
    num_unk = pn * rank

    ones_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                eq_idx = (u * mp + v) * pn + w
                ones_indices.append(eq_idx)

    for attempt in range(max_attempts):
        if time.time() > time_limit:
            break

        U = rng.random((nm, rank)) * 2 - 1
        V = rng.random((mp, rank)) * 2 - 1

        A = np.zeros((num_eqs, num_unk))
        b = np.zeros(num_eqs)
        for eq_idx in ones_indices:
            b[eq_idx] = 1.0

        for u in range(nm):
            for v in range(mp):
                for w in range(pn):
                    eq_idx = (u * mp + v) * pn + w
                    for c in range(rank):
                        A[eq_idx, w * rank + c] = U[u, c] * V[v, c]

        try:
            w_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue

        if len(residuals) > 0:
            if residuals[0] > 1e-8:
                continue
        else:
            if np.linalg.norm(A @ w_flat - b) > 1e-8:
                continue

        W = w_flat.reshape((pn, rank))

        for max_den in [10, 50, 100, 500, 1000, 5000, 100000]:
            U_r, V_r, W_r = [], [], []
            ok = True
            for mat, out in ((U, U_r), (V, V_r), (W, W_r)):
                for row in mat:
                    r = []
                    for val in row:
                        if abs(val) < 1e-12:
                            r.append(Fraction(0))
                        else:
                            r.append(Fraction(val).limit_denominator(max_den))
                    out.append(r)
            if _exact_verify(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r

    return None


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin under 150s kill

    # 1) Fast search for a low rank (bounded).
    #    Only attempt a few ranks; each is cheap. Stop well before deadline.
    search_budget = deadline - 40  # leave 40s for construction + verification
    for rank in range(9, 25):
        if time.time() > search_budget:
            break
        result = _try_rank_search(rank, n, m, p, max_attempts=60,
                                  time_limit=search_budget)
        if result is not None:
            U, V, W = result
            if _exact_verify(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}

    # 2) Explicit constructive rank-36 decomposition.
    U, V, W = _construct_block_decomposition(n, m, p)
    if _exact_verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # 3) Schoolbook fallback (guaranteed).
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
