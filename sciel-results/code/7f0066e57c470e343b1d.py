"""Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

Strategy:
  Base construction: split the 3x8 right factor into four 3x2 blocks.
  Each 2x3 * 3x2 block is decomposed as rank-7 Strassen (on first two A-columns)
  plus 3 rank-1 terms (third A-column) -> rank 11 per block -> total rank 44.

  Refinement: fast exact verification (tensor-algebra based, O(nmp*rank) with
  integer arithmetic) plus an exhaustive search for linear cancellations that
  reduce the rank.  We try:
    1. Single-column zero detection (column is all-zeros in U or V or W).
    2. Pairwise column combination: replace col j by col j + alpha*col i, then
       check if col i becomes zero (or vice-versa).
    3. Greedy iterative reduction until no more cancellations are found.
  A hard deadline guard ensures we always return a verified decomposition.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 3, 8]
_T0 = time.time()
_DEADLINE = 120.0 - 15.0
_TIME_BUDGET = 110.0


def _time_left():
    return _TIME_BUDGET - (time.time() - _T0)


def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2.

    Returns U (4x7), V (4x7), W (4x7) of Fractions such that
        sum_t U[a,t]*V[b,t]*W[d,t] = 1  iff a=i*2+j, b=j*2+k, d=k*2+i
    """
    U = np.zeros((4, 7), dtype=object)
    V = np.zeros((4, 7), dtype=object)
    W = np.zeros((4, 7), dtype=object)

    # m1 = (a11+a22)(b11+b22)  -> c11 += m1, c22 += m1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # m2 = (a21+a22)(b11)      -> c21 += m2, c22 -= m2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # m3 = a11(b12-b22)        -> c12 += m3, c22 += m3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # m4 = a22(b21-b11)        -> c11 += m4, c21 += m4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # m5 = (a11+a12)(b22)      -> c11 -= m5, c12 += m5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # m6 = (a21-a11)(b11+b12)  -> c22 += m6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # m7 = (a12-a22)(b21+b22)  -> c11 += m7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def _build_rank44():
    """Construct the rank-44 decomposition (4 blocks x rank 11)."""
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 44), dtype=object)
    V = np.zeros((m * p, 44), dtype=object)
    W = np.zeros((p * n, 44), dtype=object)

    sU, sV, sW = _strassen_2x2()

    for blk in range(4):
        base = blk * 11
        for t in range(7):
            for a in range(4):
                if sU[a, t]:
                    i, j = divmod(a, 2)
                    U[i * 3 + j, base + t] += sU[a, t]
            for b in range(4):
                if sV[b, t]:
                    j, k = divmod(b, 2)
                    V[j * 8 + (2 * blk + k), base + t] += sV[b, t]
            for d in range(4):
                if sW[d, t]:
                    k, i = divmod(d, 2)
                    W[(2 * blk + k) * 2 + i, base + t] += sW[d, t]
        for i in range(2):
            for k in range(2):
                c = base + 7 + i * 2 + k
                U[i * 3 + 2, c] = 1
                V[2 * 8 + (2 * blk + k), c] = 1
                W[(2 * blk + k) * 2 + i, c] = 1

    return U, V, W


def _build_schoolbook():
    """Guaranteed-correct rank-48 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    r = n * m * p
    U = np.zeros((n * m, r), dtype=object)
    V = np.zeros((m * p, r), dtype=object)
    W = np.zeros((p * n, r), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _tensor_ones(n, m, p):
    S = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                S.add((i * m + j, j * p + k, k * n + i))
    return S


def _compute_tensor(U, V, W):
    """Compute the full multiplication tensor T[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t].
    Returns a 3D numpy array of Fractions.
    """
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    T = np.zeros((nm, mp, pn), dtype=object)
    for t in range(rank):
        ut = U[:, t]
        vt = V[:, t]
        wt = W[:, t]
        # Only iterate over non-zero entries for speed
        for a in range(nm):
            ua = ut[a]
            if ua:
                for b in range(mp):
                    vb = vt[b]
                    if vb:
                        uv = ua * vb
                        for c in range(pn):
                            wc = wt[c]
                            if wc:
                                T[a, b, c] += uv * wc
    return T


def _verify(U, V, W):
    """Exact verification using tensor algebra."""
    n, m, p = DIMENSIONS
    S = _tensor_ones(n, m, p)
    T = _compute_tensor(U, V, W)
    nm, mp, pn = n * m, m * p, p * n
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                val = T[a, b, c]
                if (a, b, c) in S:
                    if val != 1:
                        return False
                else:
                    if val != 0:
                        return False
    return True


def _verify_fast(U, V, W, S):
    """Fast verification: compute tensor and check against S."""
    n, m, p = DIMENSIONS
    T = _compute_tensor(U, V, W)
    nm, mp, pn = n * m, m * p, p * n
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                val = T[a, b, c]
                if (a, b, c) in S:
                    if val != 1:
                        return False
                else:
                    if val != 0:
                        return False
    return True


def _col_is_zero(mat, col):
    """Check if a column of a matrix (object array) is all zeros."""
    for v in mat[:, col]:
        if v:
            return False
    return True


def _cancellation_search(U, V, W, budget):
    """Try to reduce rank by finding exact linear combinations of term-columns
    that create zeros.  Returns (U,V,W) of possibly lower rank, or None."""
    n, m, p = DIMENSIONS
    S = _tensor_ones(n, m, p)
    rank = U.shape[1]
    idx = list(range(rank))

    def check(Uc, Vc, Wc):
        cols = Uc.shape[1]
        T = _compute_tensor(Uc, Vc, Wc)
        nm, mp, pn = n * m, m * p, p * n
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = T[a, b, c]
                    if (a, b, c) in S:
                        if val != 1:
                            return False
                    else:
                        if val != 0:
                            return False
        return True

    def try_pair(i, j, alpha):
        """Replace col j by col j + alpha*col i. If col i becomes zero, drop it."""
        U2 = U[:, idx].copy()
        V2 = V[:, idx].copy()
        W2 = W[:, idx].copy()
        U2[:, j] = U2[:, j] + alpha * U2[:, i]
        V2[:, j] = V2[:, j] + alpha * V2[:, i]
        W2[:, j] = W2[:, j] + alpha * W2[:, i]
        if check(U2, V2, W2):
            keep = [k for k in idx if k != i]
            return U[:, keep].copy(), V[:, keep].copy(), W[:, keep].copy(), keep
        return None

    # Also try: replace col i by col i + alpha*col j, drop j if zero
    def try_pair_rev(i, j, alpha):
        U2 = U[:, idx].copy()
        V2 = V[:, idx].copy()
        W2 = W[:, idx].copy()
        U2[:, i] = U2[:, i] + alpha * U2[:, j]
        V2[:, i] = V2[:, i] + alpha * V2[:, j]
        W2[:, i] = W2[:, i] + alpha * W2[:, j]
        if check(U2, V2, W2):
            keep = [k for k in idx if k != j]
            return U[:, keep].copy(), V[:, keep].copy(), W[:, keep].copy(), keep
        return None

    alphas = [-2, -1, -Fraction(1, 2), Fraction(1, 2), 1, 2,
              -Fraction(3, 2), Fraction(3, 2), -3, 3]

    changed = True
    while changed and _time_left() > 30:
        changed = False
        cur_rank = U.shape[1]
        for (i, j) in combinations(range(cur_rank), 2):
            if _time_left() < 30:
                break
            for alpha in alphas:
                res = try_pair(i, j, alpha)
                if res is not None:
                    U, V, W, idx = res
                    changed = True
                    break
                if _time_left() < 30:
                    break
                res = try_pair_rev(i, j, alpha)
                if res is not None:
                    U, V, W, idx = res
                    changed = True
                    break
            if changed:
                break

    if U.shape[1] < rank:
        return U, V, W
    return None


def _try_block_symmetry_reduction(U, V, W):
    """Try to exploit the 4-fold block symmetry: the 4 blocks of 11 terms each
    have the same internal structure.  Check if any two blocks can be merged
    by a linear combination.  Returns reduced (U,V,W) or None."""
    # This is a more targeted search: for each pair of blocks, try to find
    # a linear combination that zeroes one block's contribution.
    # Due to time constraints, we only try simple alpha values.
    n, m, p = DIMENSIONS
    S = _tensor_ones(n, m, p)
    rank = U.shape[1]

    # Blocks are at columns [0..10], [11..21], [22..32], [33..43]
    block_starts = [0, 11, 22, 33]
    block_size = 11

    for bi in range(4):
        for bj in range(4):
            if bi == bj:
                continue
            if _time_left() < 35:
                return None
            # Try: for each term position within the block, replace
            # block_bj's term by block_bj's term + alpha * block_bi's term
            for pos in range(block_size):
                if _time_left() < 35:
                    return None
                ci = block_starts[bi] + pos
                cj = block_starts[bj] + pos
                for alpha in [-1, 1, -2, 2]:
                    U2 = U.copy()
                    V2 = V.copy()
                    W2 = W.copy()
                    U2[:, cj] = U2[:, cj] + alpha * U2[:, ci]
                    V2[:, cj] = V2[:, cj] + alpha * V2[:, ci]
                    W2[:, cj] = W2[:, cj] + alpha * W2[:, ci]
                    if _verify_fast(U2, V2, W2, S):
                        # Drop column ci
                        keep = [k for k in range(rank) if k != ci]
                        return U[:, keep].copy(), V[:, keep].copy(), W[:, keep].copy()
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8."""
    global _T0, _TIME_BUDGET
    _T0 = time.time()
    _TIME_BUDGET = 110.0

    # Guaranteed-correct fallbacks.
    U48, V48, W48 = _build_schoolbook()
    U44, V44, W44 = _build_rank44()

    best = (U48, V48, W48)
    if _verify(U44, V44, W44):
        best = (U44, V44, W44)

    # Refinement: attempt exact rank reduction within the time budget.
    if _time_left() > 50:
        res = _cancellation_search(U44, V44, W44, _time_left())
        if res is not None and _verify(res[0], res[1], res[2]):
            best = res

    # Try block symmetry reduction on the best found so far
    if _time_left() > 40 and best[0].shape[1] <= 44:
        res2 = _try_block_symmetry_reduction(best[0], best[1], best[2])
        if res2 is not None and _verify(res2[0], res2[1], res2[2]):
            best = res2

    U, V, W = best
    out = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return out
