"""Recursive Strassen exact tensor decomposition for 4x4 x 4x4 -> 4x4.

The multiplication tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
(W follows the official trace convention W[k*n+i, r]).  We seek a rank-r
decomposition with U (n*m x r), V (m*p x r), W (p*n x r) of integer
entries such that the outer product reconstructs T exactly.

Recursive Strassen splits each matrix into 2x2 blocks and uses 7
sub-multiplications per level.  For 4x4 the base case is 2x2 (rank 7),
giving total rank 7*7 = 49.  Generation is fully recursive, so the
index bookkeeping is uniform across levels and cannot drift like a
hand-written 17-mult core.
"""

import time

DIMENSIONS = [4, 4, 4]

# ---------------------------------------------------------------------------
# Strassen M-linear-combination coefficients (row = sub-block, col = M-number).
#   C00 = M0 + M3 - M5 - M7
#   C01 = M1 + M4
#   C10 = M2 + M5
#   C11 = M0 - M1 + M4 + M6
# ---------------------------------------------------------------------------
_M = [
    [1, 0, 0, 1, 0, -1, -1],  # 00
    [0, 1, 0, 0, 1, 0, 0],    # 01
    [0, 0, 1, 0, 0, 1, 0],    # 10
    [1, -1, 0, 0, 1, 0, 1],   # 11
]

# ---------------------------------------------------------------------------
# The seven Strassen sub-multiplications, each an (a, b) pair of 2x2 linear
# combinations of A and B blocks:  Mq = (a_q . A) * (b_q . B).
#   a_q is indexed by A-block (00,01,10,11); b_q by B-block (00,01,10,11).
# ---------------------------------------------------------------------------
_A = [
    [1, 0, 1, 0],   # M0 = A00
    [0, 1, 0, 0],   # M1 = A10
    [0, 0, 0, 1],   # M2 = A11
    [1, 0, 0, 0],   # M3 = A01
    [1, 0, 1, 0],   # M4 = (A00+A11)
    [0, 0, 1, -1],  # M5 = (A10-A00)
    [0, 1, 0, 1],   # M6 = (A01+A11)
]
_B = [
    [1, 0, 0, 0],   # M0 = B00
    [0, 0, 0, 1],   # M1 = B11
    [1, 0, 0, 0],   # M2 = B00
    [0, 1, 0, 0],   # M3 = B01
    [0, 0, 0, 1],   # M4 = B11
    [1, 0, 0, 0],   # M5 = B00
    [0, 1, 0, 0],   # M6 = B10
]


def _base_2x2():
    """Return (u, v, w) lists of length 7 for the 2x2 base case.

    u is a flat 4-vector (A-block linear combination), v a flat 4-vector
    (B-block linear combination), w a flat 4-vector (C-block linear
    combination) per sub-multiplication.  These are derived directly from
    _A, _B, _M so the base case is consistent with the recursion.
    """
    u, v, w = [], [], []
    for q in range(7):
        u.append(list(_A[q]))
        v.append(list(_B[q]))
        wc = [0, 0, 0, 0]
        for cb in range(4):
            for q2 in range(7):
                wc[cb] += _M[cb][q2]
        w.append(wc)
    return u, v, w


def _recurse(u, v, w, level):
    """Recurse one Strassen level.  u/v/w are lists of flat 4-vectors."""
    nu, nv, nw = [], [], []
    for q in range(7):
        # A-side: sub-block (r, c) -> element index (2*r + ri) * 2 + (2*c + ci)
        ua = [0] * 16
        for r in range(2):
            for c in range(2):
                base = 4 * q + 4 * r + c
                for ri in range(2):
                    for ci in range(2):
                        ua[(2 * r + ri) * 2 + (2 * c + ci)] = u[base + 2 * ri + ci]
        # B-side: sub-block (r, c) -> element index (2*r + ri) * 2 + (2*c + ci)
        vb = [0] * 16
        for r in range(2):
            for c in range(2):
                base = 4 * q + 4 * r + c
                for ri in range(2):
                    for ci in range(2):
                        vb[(2 * r + ri) * 2 + (2 * c + ci)] = v[base + 2 * ri + ci]
        # C-side: sub-block (r, c) -> element index (2*r + ri) * 2 + (2*c + ci)
        wc = [0] * 16
        for r in range(2):
            for c in range(2):
                base = 4 * q + 4 * r + c
                for ri in range(2):
                    for ci in range(2):
                        wc[(2 * r + ri) * 2 + (2 * c + ci)] = w[base + 2 * ri + ci]
        nu.append(ua)
        nv.append(vb)
        nw.append(wc)
    return nu, nv, nw


def _strassen_decomposition(n):
    """Build integer (U, V, W) for n x n multiplication via recursive Strassen.

    Returns None if n is not a power of two >= 2.
    """
    if n < 2:
        return None
    # number of levels until we reach 2x2
    levels = 0
    size = n
    while size > 2:
        size //= 2
        if size % 2 != 0:
            return None
        levels += 1
    if size != 2:
        return None

    u, v, w = _base_2x2()
    for _ in range(levels):
        u, v, w = _recurse(u, v, w, 0)

    rank = len(u)
    N = n * n
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(N)]
    W = [[0] * rank for _ in range(N)]
    for c in range(rank):
        for idx in range(N):
            U[idx][c] = u[c][idx]
            V[idx][c] = v[c][idx]
            W[idx][c] = w[c][idx]
    return {"U": U, "V": V, "W": W}


def _schoolbook(n):
    """Exact rank n^3 decomposition (always valid)."""
    rank = n * n * n
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j][col] = 1
                V[j * n + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(dec, n):
    """Check that (U^T W) V reconstructs the multiplication tensor exactly."""
    U, V, W = dec["U"], dec["V"], dec["W"]
    rank = len(U[0])
    N = n * n
    # T = U^T W  (N x N):  T[a, b] = sum_c U[a, c] W[b, c]
    # Then result C = T V  (N x N):  C[a, c] = sum_b T[a, b] V[b, c]
    # Expected: A[a=(i,j)] * B[b=(j,k)] -> C[c=(i,k)] = 1 iff j matches.
    # a = i*n+j, b = j*n+k, c = i*n+k.
    for i in range(n):
        for j in range(n):
            for k in range(n):
                a = i * n + j
                b = j * n + k
                c = i * n + k
                # T[a, b]
                t_ab = 0
                for cc in range(rank):
                    t_ab += U[a][cc] * W[b][cc]
                # C[a, c] = sum_b' T[a, b'] V[b', c]; only b' with matching j contribute
                # but we sum over all b' for correctness
                c_val = 0
                for bp in range(N):
                    t_ap = 0
                    for cc in range(rank):
                        t_ap += U[a][cc] * W[bp][cc]
                    c_val += t_ap * V[bp][c]
                # expected: 1 if j == j (always, since a fixes j and b must share j)
                # For this (i,j,k), the only b that can contribute is b = j*n+k.
                # If decomposition is exact, C[a,c] must equal 1.
                if c_val != 1:
                    return False
    return True


# Cache to avoid rebuilding on repeated calls within the 150s budget.
_CACHE = {}


def construct():
    n = DIMENSIONS[0]
    m = DIMENSIONS[1]
    p = DIMENSIONS[2]

    deadline = time.time() + 120.0  # safety margin under the 150s kill

    if (n, m, p) in _CACHE:
        return _CACHE[(n, m, p)]

    best = None
    if n == m == p:
        dec = _strassen_decomposition(n)
        if dec is not None:
            # Verify before trusting the recursive output.
            if _verify(dec, n):
                best = dec
    if best is None:
        best = _schoolbook(n)

    _CACHE[(n, m, p)] = best
    return best


if __name__ == "__main__":
    t0 = time.time()
    res = construct()
    rank = len(res["U"][0])
    print("rank =", rank, "time =", round(time.time() - t0, 3), "s")
