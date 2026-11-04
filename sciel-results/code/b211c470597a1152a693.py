"""Matrix multiplication 3,5,6 - rank-minimal tensor decomposition.

Strategy: block decomposition using Strassen for 2x2x2 sub-blocks, with an
exhaustive search over all valid tilings of n=3, m=5, p=6.

Key insight: the 3x5x6 tensor can be tiled in many ways. Each 2x2x2 block
uses Strassen (rank 7 instead of 8), while other blocks use schoolbook
(rank = product of dimensions). We search all tilings and verify each one.
"""
import time
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

# ---------------------------------------------------------------------------
# Exact verification helpers
# ---------------------------------------------------------------------------
def _verify_int(U, V, W, n, m, p):
    """Fast integer verification: every (i,j,k) must sum to exactly 1."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True


def _verify_frac(U, V, W, n, m, p):
    """Exact Fraction verification (handles half-integers)."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if s != 1:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p):
    """Dispatch to integer verification if all entries are whole, else Fraction."""
    is_int = all(float(x) == int(x) for row in U for x in row) and \
             all(float(x) == int(x) for row in V for x in row) and \
             all(float(x) == int(x) for row in W for x in row)
    if is_int:
        return _verify_int(U, V, W, n, m, p)
    return _verify_frac(U, V, W, n, m, p)


# ---------------------------------------------------------------------------
# Strassen 2x2x2 block
# ---------------------------------------------------------------------------
def _strassen_2x2x2():
    """Return (U,V,W) lists for a 2x2x2 Strassen decomposition (rank 7).

    Convention: U[i*2+j, c] -> A[i,j]; V[j*2+k, c] -> B[j,k]; W[k*2+i, c] -> C[i,k].
    """
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]

    # m1 = (a+d)(e+h)  -> C[0,0]+=1, C[1,1]+=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    # m2 = (c+d)e      -> C[1,0]+=1, C[1,1]+=-1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1
    # m3 = a(f-h)      -> C[0,1]+=1, C[1,1]+=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1; W[3][2] = 1
    # m4 = d(g-e)      -> C[0,0]+=1, C[1,0]+=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1
    # m5 = (a+b)h      -> C[0,0]+=-1, C[0,1]+=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[2][4] = 1
    # m6 = (c-a)(e+f)  -> C[1,1]+=1
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    # m7 = (b-d)(g+h)  -> C[0,0]+=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    return U, V, W


# ---------------------------------------------------------------------------
# Block decomposition builder
# ---------------------------------------------------------------------------
def _build_block(n, m, p, n_sizes, m_sizes, p_sizes):
    """Build full factor matrices via block tiling; 2x2x2 blocks use Strassen."""
    n_off = []
    o = 0
    for s in n_sizes:
        n_off.append(o); o += s
    m_off = []
    o = 0
    for s in m_sizes:
        m_off.append(o); o += s
    p_off = []
    o = 0
    for s in p_sizes:
        p_off.append(o); o += s

    # total rank
    total = 0
    for ni in n_sizes:
        for mj in m_sizes:
            for pk in p_sizes:
                if ni == 2 and mj == 2 and pk == 2:
                    total += 7
                else:
                    total += ni * mj * pk

    U = [[0] * total for _ in range(n * m)]
    V = [[0] * total for _ in range(m * p)]
    W = [[0] * total for _ in range(p * n)]

    col = 0
    for ii, ni in enumerate(n_sizes):
        i0 = n_off[ii]
        for jj, mj in enumerate(m_sizes):
            j0 = m_off[jj]
            for kk, pk in enumerate(p_sizes):
                k0 = p_off[kk]
                if ni == 2 and mj == 2 and pk == 2:
                    Ub, Vb, Wb = _strassen_2x2x2()
                    r = 7
                else:
                    r = ni * mj * pk
                    Ub = [[0] * r for _ in range(ni * mj)]
                    Vb = [[0] * r for _ in range(mj * pk)]
                    Wb = [[0] * r for _ in range(pk * ni)]
                    c = 0
                    for li in range(ni):
                        for lj in range(mj):
                            for lk in range(pk):
                                Ub[li * mj + lj][c] = 1
                                Vb[lj * pk + lk][c] = 1
                                Wb[lk * ni + li][c] = 1
                                c += 1
                for li in range(ni):
                    for lj in range(mj):
                        U[(i0 + li) * m + (j0 + lj)][col:col + r] = Ub[li * mj + lj]
                for lj in range(mj):
                    for lk in range(pk):
                        V[(j0 + lj) * p + (k0 + lk)][col:col + r] = Vb[lj * pk + lk]
                for lk in range(pk):
                    for li in range(ni):
                        W[(k0 + lk) * n + (i0 + li)][col:col + r] = Wb[lk * ni + li]
                col += r
    return U, V, W, total


def _estimate_rank(n_sizes, m_sizes, p_sizes):
    """Estimate the rank of a tiling without building the full matrices."""
    total = 0
    for ni in n_sizes:
        for mj in m_sizes:
            for pk in p_sizes:
                if ni == 2 and mj == 2 and pk == 2:
                    total += 7
                else:
                    total += ni * mj * pk
    return total


def _all_compositions(n, max_part=None):
    """Generate all compositions of n into positive integers, optionally with max part."""
    if max_part is None:
        max_part = n
    def _comp(remaining, max_p):
        if remaining == 0:
            yield []
        else:
            for k in range(1, min(remaining, max_p) + 1):
                for rest in _comp(remaining - k, max_p):
                    yield [k] + rest
    return list(_comp(n, max_part))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s wall-clock

    n, m, p = DIMENSIONS  # 3, 5, 6

    # Generate all possible tilings
    n_tilings = _all_compositions(n, 3)
    m_tilings = _all_compositions(m, 3)
    p_tilings = _all_compositions(p, 3)

    # Sort tilings by estimated rank to find good ones quickly
    all_candidates = []
    for nt in n_tilings:
        for mt in m_tilings:
            for pt in p_tilings:
                est = _estimate_rank(nt, mt, pt)
                all_candidates.append((est, nt, mt, pt))

    # Sort by estimated rank
    all_candidates.sort(key=lambda x: x[0])

    best = None
    best_rank = float('inf')

    for est, nt, mt, pt in all_candidates:
        if time.time() > deadline:
            break
        
        # Skip if this tiling can't beat the current best
        if est >= best_rank:
            continue

        U_t, V_t, W_t, rank_t = _build_block(n, m, p, nt, mt, pt)
        if _verify_fast(U_t, V_t, W_t, n, m, p):
            if rank_t < best_rank:
                best_rank = rank_t
                best = (rank_t, U_t, V_t, W_t)

    if best is None:
        # Fallback to schoolbook (rank 90)
        rank90 = n * m * p
        U = [[0] * rank90 for _ in range(n * m)]
        V = [[0] * rank90 for _ in range(m * p)]
        W = [[0] * rank90 for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        best = (rank90, U, V, W)

    return {"U": best[1], "V": best[2], "W": best[3]}
