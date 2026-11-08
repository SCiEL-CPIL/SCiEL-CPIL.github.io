import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    # (U,V,W) each 4x7; U row = i*2+j (A 2x2), V row = j*2+k, W row = k*2+i
    U = [[0] * 7 for _ in range(4)]
    V = [[0] * 7 for _ in range(4)]
    W = [[0] * 7 for _ in range(4)]
    defs = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (0, 1, -1), (1, 0, -1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(1, 0, 1), (0, 0, -1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    ]
    for t, (ue, ve, we) in enumerate(defs):
        for (i, j, c) in ue:
            U[i * 2 + j][t] += c
        for (j, k, c) in ve:
            V[j * 2 + k][t] += c
        for (k, i, c) in we:
            W[k * 2 + i][t] += c
    return U, V, W


def _build_4x4x4():
    # 2-level Strassen: (4x4)*(4x4) rank 49
    U = [[0] * 49 for _ in range(16)]
    V = [[0] * 49 for _ in range(16)]
    W = [[0] * 49 for _ in range(16)]
    U2, V2, W2 = _strassen_2x2()
    outer = [
        ([(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)], [(0, 0, 1), (1, 1, 1)]),
        ([(1, 0, 1), (1, 1, 1)], [(0, 0, 1)], [(1, 0, 1), (1, 1, -1)]),
        ([(0, 0, 1)], [(0, 1, 1), (1, 1, -1)], [(0, 1, 1), (1, 1, 1)]),
        ([(1, 1, 1)], [(1, 0, 1), (0, 0, -1)], [(1, 0, 1), (0, 0, -1)]),
        ([(0, 0, 1), (0, 1, 1)], [(1, 1, 1)], [(0, 0, -1), (0, 1, 1)]),
        ([(1, 0, 1), (0, 0, -1)], [(0, 0, 1), (0, 1, 1)], [(1, 1, 1)]),
        ([(0, 1, 1), (1, 1, -1)], [(1, 0, 1), (1, 1, 1)], [(0, 0, 1)]),
    ]
    for s, (ue, ve, we) in enumerate(outer):
        for t in range(7):
            col = s * 7 + t
            for (bi, bj, c) in ue:
                for i in range(2):
                    for j in range(2):
                        gi = bi * 2 + i
                        gj = bj * 2 + j
                        cc = U2[i * 2 + j][t]
                        if cc:
                            U[gi * 4 + gj][col] += cc * c
            for (bj, bk, c) in ve:
                for j in range(2):
                    for k in range(2):
                        gj = bj * 2 + j
                        gk = bk * 2 + k
                        cc = V2[j * 2 + k][t]
                        if cc:
                            V[gj * 4 + gk][col] += cc * c
            for (bk, bi, c) in we:
                for k in range(2):
                    for i in range(2):
                        gk = bk * 2 + k
                        gi = bi * 2 + i
                        cc = W2[k * 2 + i][t]
                        if cc:
                            W[gk * 4 + gi][col] += cc * c
    return U, V, W


def _build_outer(n, m, p):
    # (n x m) * (m x p) where m is the shared index; rank n*m*p (schoolbook)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def _combine(parts, n, m, p):
    total = sum(len(d[0][0]) for d in parts)
    U = [[0] * total for _ in range(n * m)]
    V = [[0] * total for _ in range(m * p)]
    W = [[0] * total for _ in range(p * n)]
    off = 0
    for (Us, Vs, Ws, (i0, j0, ii, jj, k0, kk)) in parts:
        nA = ii
        mA = jj
        nB = kk
        for r in range(nA * mA):
            i = r // mA
            j = r % mA
            for c in range(len(Us[0])):
                U[(i0 + i) * m + (j0 + j)][off + c] = Us[r][c]
        for r in range(mA * nB):
            j = r // nB
            k = r % nB
            for c in range(len(Vs[0])):
                V[(j0 + j) * p + (k0 + k)][off + c] = Vs[r][c]
        for r in range(nB * nA):
            k = r // nA
            i = r % nA
            for c in range(len(Ws[0])):
                W[(k0 + k) * n + (i0 + i)][off + c] = Ws[r][c]
        off += len(Us[0])
    return U, V, W


def _reduce_rank(U, V, W, n, m, p):
    # Rank-reduction: the decomposition tensor T (rank R) lives in the span of
    # its R vectors in R^(n*m*p). If those vectors have linear dependencies,
    # T has rank < R. Project the factor matrices onto a basis of that span to
    # obtain an equivalent minimal-rank decomposition.
    R = len(U[0])
    if R == 0:
        return U, V, W
    import numpy as np
    uflat = np.array([U[i * m + j][t] for t in range(R) for i in range(n) for j in range(m)], dtype=float)
    vflat = np.array([V[j * p + k][t] for t in range(R) for j in range(m) for k in range(p)], dtype=float)
    wflat = np.array([W[k * n + i][t] for t in range(R) for k in range(p) for i in range(n)], dtype=float)
    T = np.stack([uflat, vflat, wflat], axis=1)  # shape (N, R)
    # Orthonormal basis of the row space of T
    _, svals, Vt = np.linalg.svd(T, full_matrices=False)
    r = int(np.sum(svals > 1e-8))
    if r >= R:
        return U, V, W  # no reduction available
    B = Vt[:r]  # (r, R)
    Unew = np.array(U, dtype=float).reshape(-1, R) @ B.T
    Vnew = np.array(V, dtype=float).reshape(-1, R) @ B.T
    Wnew = np.array(W, dtype=float).reshape(-1, R) @ B.T
    # Round to nearest 1/8 (entries are small integer/half-integer combos)
    def _round(M):
        return np.round(M * 8.0) / 8.0
    Unew = _round(Unew).reshape(-1, r)
    Vnew = _round(Vnew).reshape(-1, r)
    Wnew = _round(Wnew).reshape(-1, r)
    Uout = Unew.tolist()
    Vout = Vnew.tolist()
    Wout = Wnew.tolist()
    return Uout, Vout, Wout


def _verify(U, V, W, n, m, p, trials=4, seed=12345):
    import random
    rng = random.Random(seed)
    rank = len(U[0])
    for _ in range(trials):
        A = [[rng.randint(-4, 4) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(-4, 4) for _ in range(p)] for _ in range(m)]
        C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        for i in range(n):
            for k in range(p):
                val = 0.0
                for t in range(rank):
                    au = sum(U[i * m + j][t] * A[i][j] for j in range(m))
                    bu = sum(V[j * p + k][t] * B[j][k] for j in range(m))
                    val += W[k * n + i][t] * au * bu
                if abs(val - C[i][k]) > 1e-6:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return U, V, W


def construct():
    start = time.time()
    n, m, p = DIMENSIONS
    deadline = start + 115

    def _elapsed():
        return time.time() - start

    try:
        # Tiling:
        #   4x4x4 (Strassen rank 49): A[0:4,0:4] * B[0:4,0:6]
        #   4x1x6 (outer rank 24)  : A[0:4,4]   * B[4,0:6]
        #   4x1x2 (outer rank 8)   : A[0:4,4]   * B[4,4:6]
        # Total naive rank = 49 + 24 + 8 = 81
        U4, V4, W4 = _build_4x4x4()
        U1, V1, W1 = _build_outer(4, 1, 6)
        U2, V2, W2 = _build_outer(4, 1, 2)

        parts = [
            (U4, V4, W4, (0, 0, 4, 4, 0, 6)),
            (U1, V1, W1, (0, 4, 4, 1, 4, 6)),
            (U2, V2, W2, (0, 4, 4, 1, 4, 2)),
        ]
        U, V, W = _combine(parts, n, m, p)

        if _elapsed() > deadline - 25:
            U, V, W = _schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}

        # Validate the naive tiling
        if not _verify(U, V, W, n, m, p, trials=3):
            U, V, W = _schoolbook(n, m, p)
            return {"U": U, "V": V, "W": W}

        # Structural refinement: rank reduction via SVD of the tensor core
        if _elapsed() < deadline - 12:
            try:
                Ur, Vr, Wr = _reduce_rank(U, V, W, n, m, p)
                if _verify(Ur, Vr, Wr, n, m, p, trials=4):
                    U, V, W = Ur, Vr, Wr
            except Exception:
                pass  # keep the naive tiling

        if _elapsed() > deadline - 5:
            return {"U": U, "V": V, "W": W}

        if _verify(U, V, W, n, m, p, trials=4):
            return {"U": U, "V": V, "W": W}

        # Fallback
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

    except Exception:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
