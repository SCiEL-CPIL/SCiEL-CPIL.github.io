"""Hybrid Strassen+schoolbook tiling with packing search for 3x5 x 5x7."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START

# Strassen terms for 2x2 x 2x2. Each term: (A_factors, B_factors, C_factors)
# A_factors: list of (row, col, coeff) within the 2x2 A block (row=i, col=j)
# B_factors: list of (row, col, coeff) within the 2x2 B block (row=j, col=k)
# C_factors: list of (row, col, coeff) within the 2x2 C block (row=i, col=k)
# Standard Strassen:
#   M1 = (A11+A22)(B11+B22)
#   M2 = (A21+A22) B11
#   M3 = A11 (B12-B22)
#   M4 = A22 (B21-B11)
#   M5 = (A11+A12) B22
#   M6 = (A21-A11)(B11+B12)
#   M7 = (A12-A22)(B21+B22)
#   C11 = M1 + M4 - M5 + M7
#   C12 = M3 + M5
#   C21 = M2 + M4
#   C22 = M1 - M2 + M3 + M6
_STRASSEN = [
    # M1
    ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
    # M2
    ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
    # M3
    ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
    # M4
    ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
    # M5
    ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
    # M6
    ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
    # M7
    ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
]

# Precompute the 7x4x4x4 tensor of Strassen coefficients:
# S[r, i, j, k] such that for a 2x2x2 block, the contribution to T[i,j,k]
# (where i is A-row, j is A-col/B-row, k is B-col) equals sum_r U[i,j]V[j,k]W[k,i].
# We build U,V,W per term and read off the (i,j,k) coefficient.
def _build_strassen_tensor():
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    for r, (af, bf, cf) in enumerate(_STRASSEN):
        for (ar, ac, c) in af:
            U[ar*2 + ac, r] += c
        for (br, bc, c) in bf:
            V[br*2 + bc, r] += c
        for (cr, cc, c) in cf:
            # C row = i = cr, C col = k = cc. W indexed by (k, i) -> row = cc*2 + cr
            W[cc*2 + cr, r] += c
    S = np.zeros((2, 2, 2, 7))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                S[i, j, k] = U[i*2 + j] * V[j*2 + k] * W[k*2 + i]
    return S

_S = _build_strassen_tensor()  # shape (2,2,2,7)


def _partitions(dim, max_size=4):
    """All ordered compositions of dim into parts of size <= max_size (allow 1..max_size)."""
    if dim <= 0:
        return [()]
    results = []
    def rec(remaining, acc):
        if remaining == 0:
            results.append(tuple(acc))
            return
        for s in range(1, max_size + 1):
            if s <= remaining:
                acc.append(s)
                rec(remaining - s, acc)
                acc.pop()
    rec(dim, [])
    return results


def _rank_of_packing(gi, gj, gk):
    """Total rank for a given tiling (list of block sizes per dimension)."""
    rank = 0
    for si in gi:
        for sj in gj:
            for sk in gk:
                if si == 2 and sj == 2 and sk == 2:
                    rank += 7
                else:
                    rank += si * sj * sk
    return rank


def _build_decomposition(gi, gj, gk, n, m, p):
    """Build full-size U, V, W for a given tiling. Returns (U, V, W, rank)."""
    n_off = [0]
    for s in gi:
        n_off.append(n_off[-1] + s)
    m_off = [0]
    for s in gj:
        m_off.append(m_off[-1] + s)
    p_off = [0]
    for s in gk:
        p_off.append(p_off[-1] + s)

    # Determine total rank
    rank = 0
    terms = []
    for ni, si in enumerate(gi):
        for mi, sj in enumerate(gj):
            for pi, sk in enumerate(gk):
                if si == 2 and sj == 2 and sk == 2:
                    terms.append((ni, mi, pi, 'strassen'))
                    rank += 7
                else:
                    terms.append((ni, mi, pi, 'trivial'))
                    rank += si * sj * sk

    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))

    col = 0
    for ni, mi, pi, method in terms:
        n0 = n_off[ni]
        m0 = m_off[mi]
        p0 = p_off[pi]
        si = gi[ni]
        sj = gj[mi]
        sk = gk[pi]

        if method == 'strassen':
            for r in range(7):
                af, bf, cf = _STRASSEN[r]
                for (ar, ac, c) in af:
                    U[(n0+ar)*m + (m0+ac), col] += c
                for (br, bc, c) in bf:
                    V[(m0+br)*p + (p0+bc), col] += c
                for (cr, cc, c) in cf:
                    W[(p0+cc)*n + (n0+cr), col] += c
                col += 1
        else:
            for ii in range(si):
                for jj in range(sj):
                    for kk in range(sk):
                        U[(n0+ii)*m + (m0+jj), col] = 1
                        V[(m0+jj)*p + (p0+kk), col] = 1
                        W[(p0+kk)*n + (n0+ii), col] = 1
                        col += 1

    return U, V, W, rank


def _verify_fast(U, V, W, n, m, p, tol=1e-8):
    """Vectorized verification: for each (i,j,k), sum_r U[i*m+j,r]V[j*p+k,r]W[k*n+i,r] == 1."""
    # Build index arrays
    i_idx = np.repeat(np.arange(n), m*p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n*m)

    a_idx = i_idx * m + j_idx          # U rows
    b_idx = j_idx * p + k_idx          # V rows
    c_idx = k_idx * n + i_idx          # W rows

    # U[a_idx, r] for all (ijk) and r
    Ua = U[a_idx, :]   # (N, rank)
    Vb = V[b_idx, :]   # (N, rank)
    Wc = W[c_idx, :]   # (N, rank)
    prod = Ua * Vb * Wc
    s = prod.sum(axis=1)
    return np.all(np.abs(s - 1.0) <= tol)


def _best_packing(n, m, p, deadline):
    """Search over tilings to find the minimum-rank valid decomposition."""
    parts_n = _partitions(n, 4)
    parts_m = _partitions(m, 4)
    parts_p = _partitions(p, 4)

    # Candidate packings sorted by predicted rank (ascending)
    candidates = []
    for gi in parts_n:
        for gj in parts_m:
            for gk in parts_p:
                r = _rank_of_packing(gi, gj, gk)
                candidates.append((r, gi, gj, gk))
    candidates.sort(key=lambda x: x[0])

    best = None
    for r, gi, gj, gk in candidates:
        if time.time() > deadline:
            break
        U, V, W, rank = _build_decomposition(gi, gj, gk, n, m, p)
        if _verify_fast(U, V, W, n, m, p):
            best = (U, V, W, rank)
            break  # sorted by rank, first valid is optimal among candidates
    return best


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W, rank


def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s limit
    n, m, p = DIMENSIONS

    best = _best_packing(n, m, p, deadline)

    if best is None:
        U, V, W, rank = _schoolbook(n, m, p)
    else:
        U, V, W, rank = best

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
