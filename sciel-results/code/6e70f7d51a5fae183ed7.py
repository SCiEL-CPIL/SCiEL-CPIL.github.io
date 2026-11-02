"""Rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.

Strategy (structural idea from the best-known solution):
  Split the 3x8 right factor into four 3x2 blocks.  Each 2x3 * 3x2 block is
  decomposed as (rank-7 Strassen on the first two A-columns) plus three rank-1
  terms for the third A-column, i.e. rank 11 per block  ->  total rank 44.

Refinement machinery added here:
  * An exact integer tensor verification harness (no floating point).
  * A time-budgeted "cancellation search" that tries to combine the 44 terms
    into fewer (exact linear dependence) to push the rank below 44.
  * A hard deadline guard: always return the best *verified* decomposition,
    defaulting to the guaranteed-correct rank-44 construction.
"""
import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 3, 8]
_T0 = time.time()
_DEADLINE = 120.0 - 15.0   # safety margin before the 120s wall-clock kill
_TIME_BUDGET = 110.0       # search budget inside construct()


def _time_left():
    return _TIME_BUDGET - (time.time() - _T0)


def _is_int(f):
    return isinstance(f, Fraction) and f.denominator == 1


def _strassen_2x2():
    """Rank-7 Strassen decomposition for 2x2 * 2x2.

    Returns U (4x7), V (4x7), W (4x7) of Fractions such that
        sum_t U[a,t]*V[b,t]*W[d,t] = 1  iff a=i*2+j, b=j*2+k, d=k*2+i
    (i.e. the shared index j matches and the output (i,k) is addressed as d=k*2+i).
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
    n, m, p = DIMENSIONS  # 2, 3, 8
    U = np.zeros((n * m, 44), dtype=object)
    V = np.zeros((m * p, 44), dtype=object)
    W = np.zeros((p * n, 44), dtype=object)

    sU, sV, sW = _strassen_2x2()

    for blk in range(4):
        base = blk * 11
        # 7 Strassen terms on the first two A-columns (j = 0,1)
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
        # 3 rank-1 terms for the third A-column (j = 2)
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
    """Set of (a,b,c) indices with value 1 for the multiplication tensor.

    a = i*m+j, b = j*p+k, c = k*n+i.
    """
    S = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                S.add((i * m + j, j * p + k, k * n + i))
    return S


def _verify(U, V, W):
    """Exact verification: sum_t U[a,t]V[b,t]W[c,t] == 1 iff (a,b,c) in tensor."""
    n, m, p = DIMENSIONS
    S = _tensor_ones(n, m, p)
    cols = U.shape[1]
    for a in range(n * m):
        ua = U[a]
        for b in range(m * p):
            vb = V[b]
            for c in range(p * n):
                wc = W[c]
                s = Fraction(0)
                for t in range(cols):
                    s += ua[t] * vb[t] * wc[t]
                    if s and (s > 1 or s < -1):
                        return False
                if (a, b, c) in S:
                    if s != 1:
                        return False
                else:
                    if s != 0:
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
        for a in range(n * m):
            ua = Uc[a]
            for b in range(m * p):
                vb = Vc[b]
                for c in range(p * n):
                    wc = Wc[c]
                    s = Fraction(0)
                    for t in range(cols):
                        s += ua[t] * vb[t] * wc[t]
                        if s and (s > 1 or s < -1):
                            return False
                    if (a, b, c) in S:
                        if s != 1:
                            return False
                    else:
                        if s != 0:
                            return False
        return True

    def try_pair(i, j, alpha):
        U2 = U[:, idx].copy()
        V2 = V[:, idx].copy()
        W2 = W[:, idx].copy()
        U2[:, j] = U2[:, j] + alpha * U2[:, i]
        V2[:, j] = V2[:, j] + alpha * V2[:, i]
        W2[:, j] = W2[:, j] + alpha * W2[:, i]
        if check(U2, V2, W2):
            # drop the zeroed column i
            keep = [k for k in idx if k != i]
            return U[:, keep].copy(), V[:, keep].copy(), W[:, keep].copy(), keep

    alphas = [-2, -1, -Fraction(1, 2), Fraction(1, 2), 1, 2]
    changed = True
    while changed and _time_left() > 30:
        changed = False
        for (i, j) in combinations(range(U.shape[1]), 2):
            if _time_left() < 30:
                break
            for alpha in alphas:
                res = try_pair(i, j, alpha)
                if res is not None:
                    U, V, W, idx = res
                    changed = True
                    break
            if changed:
                break
    if U.shape[1] < rank:
        return U, V, W
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x3 * 3x8."""
    global _T0, _TIME_BUDGET
    _T0 = time.time()
    _TIME_BUDGET = 110.0

    # Guaranteed-correct fallbacks (cheapest first).
    U48, V48, W48 = _build_schoolbook()
    U44, V44, W44 = _build_rank44()

    best = (U48, V48, W48)
    if _verify(U44, V44, W44):
        best = (U44, V44, W44)

    # Refinement: attempt an exact rank reduction within the time budget.
    if _time_left() > 40:
        res = _cancellation_search(U44, V44, W44, _time_left())
        if res is not None and _verify(res[0], res[1], res[2]):
            best = res

    U, V, W = best
    out = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    return out
