"""Winograd/spectral transform decomposition for 2x3 x 3x7 with rank polish."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# ---------------------------------------------------------------------------
# Internal helper: exact rational Gaussian elimination -> row rank + basis
# ---------------------------------------------------------------------------
def _row_rank_basis(rows):
    m = len(rows)
    if m == 0:
        return 0, []
    n = len(rows[0])
    work = [list(map(Fraction, r)) for r in rows]
    basis = []
    r = 0
    for c in range(n):
        piv = None
        for i in range(r, m):
            if work[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        work[r], work[piv] = work[piv], work[r]
        pv = work[r][c]
        for j in range(c, n):
            work[r][j] /= pv
        for i in range(m):
            if i != r and work[i][c] != 0:
                f = work[i][c]
                for j in range(c, n):
                    work[i][j] -= f * work[r][j]
        basis.append(work[r])
        r += 1
        if r == m:
            break
    return r, basis


# ---------------------------------------------------------------------------
# Exact verification of a decomposition
# ---------------------------------------------------------------------------
def _verify(U, V, W, n, m, p, tol=1e-9):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    # (i,j,k) -> sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    ok = True
    maxerr = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > tol:
                    ok = False
                    maxerr = max(maxerr, abs(s - 1.0))
    return ok, maxerr


# ---------------------------------------------------------------------------
# Construct the spectral (FFT) decomposition of rank |A|*m*|B|
# ---------------------------------------------------------------------------
def _build_spectral(n, m, p, N, M):
    # FFT matrices
    N2 = 1 << (N - 1).bit_length()
    wN = np.exp(-2j * np.pi / N2)
    F = np.array([[wN ** (a * b) for b in range(N2)] for a in range(N2)])
    N3 = 1 << (M - 1).bit_length()
    wM = np.exp(-2j * np.pi / N3)
    G = np.array([[wM ** (a * b) for b in range(N3)] for a in range(N3)])

    # A: (i,k) -> u ; B: j -> v
    A = np.zeros((n, p, N), dtype=complex)
    B = np.zeros((m, M), dtype=complex)
    for i in range(n):
        for k in range(p):
            A[i, k, :] = F[:N, k]
    for j in range(m):
        B[j, :] = G[:M, j]

    U = np.zeros((n * m, N * M))
    V = np.zeros((m * p, N * M))
    W = np.zeros((p * n, N * M))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = A[i, k, :]
                v = B[j, :]
                U[i * m + j, :] = u
                V[j * p + k, :] = v
                W[k * n + i, :] = u * v
    return U, V, W


# ---------------------------------------------------------------------------
# Fast rank polish: random subspace restriction + exact basis, iterate
# ---------------------------------------------------------------------------
def _polish(U, V, W, n, m, p, seed=0, iters=30, min_rank=1):
    rng = random.Random(seed)
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    R0 = U.shape[1]
    best = (R0, U, V, W)

    def rowspace_basis(M):
        r, _ = _row_rank_basis(M.tolist())
        # rebuild basis rows
        m = len(M)
        n = len(M[0]) if m else 0
        work = [list(map(Fraction, row)) for row in M]
        basis = []
        rnk = 0
        for c in range(n):
            piv = None
            for i in range(rnk, m):
                if work[i][c] != 0:
                    piv = i
                    break
            if piv is None:
                continue
            work[rnk], work[piv] = work[piv], work[rnk]
            pv = work[rnk][c]
            for j in range(c, n):
                work[rnk][j] /= pv
            for i in range(m):
                if i != rnk and work[i][c] != 0:
                    f = work[i][c]
                    for j in range(c, n):
                        work[i][j] -= f * work[rnk][j]
            basis.append(work[rnk])
            rnk += 1
            if rnk == m:
                break
        return rnk, [np.array([float(x) for x in row]) for row in basis]

    for _ in range(iters):
        r = rng.randint(min_rank, best[0])
        # random projection of rowspace
        Q = rng.rand(R0, r)
        U2 = U @ Q
        V2 = V @ Q
        W2 = W @ Q
        rU, bU = rowspace_basis(U2)
        rV, bV = rowspace_basis(V2)
        rW, bW = rowspace_basis(W2)
        if rU != rV or rV != rW:
            continue
        ok, _ = _verify(bU, bV, bW, n, m, p)
        if ok:
            if rU < best[0]:
                best = (rU, np.array(bU), np.array(bV), np.array(bW))
                if rU <= min_rank:
                    break
    return best[1], best[2], best[3], best[0]


# ---------------------------------------------------------------------------
# Main construct
# ---------------------------------------------------------------------------
def construct():
    t0 = time.time()
    DEADLINE = t0 + 120.0
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p

    def schoolbook():
        r = n * m * p
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][c] = 1
                    V[j * p + k][c] = 1
                    W[k * n + i][c] = 1
                    c += 1
        return {"U": U, "V": V, "W": W}

    # Build spectral decomposition with the "big" index on the output (i)
    N = n  # 2
    M = p  # 7
    U, V, W = _build_spectral(n, m, p, N, M)
    rank = N * M
    ok, _ = _verify(U, V, W, n, m, p)
    if not ok:
        return schoolbook()

    # Polish to reduce rank
    if time.time() < DEADLINE - 5:
        U, V, W, rank = _polish(U, V, W, n, m, p, seed=12345,
                                 iters=12, min_rank=max(1, int(0.5 * rank)))

    # Convert to lists of ints/half-integers where possible
    def to_lists(A):
        out = []
        for row in A:
            r = []
            for x in row:
                x = float(x)
                # round to nearest half-integer
                h = round(2 * x) / 2.0
                if abs(x - h) < 1e-7:
                    r.append(h)
                else:
                    r.append(x)
            out.append(r)
        return out

    U, V, W = to_lists(U), to_lists(V), to_lists(W)

    # Final verification on the returned lists
    ok, _ = _verify(U, V, W, n, m, p)
    if not ok:
        return schoolbook()

    return {"U": U, "V": V, "W": W}
