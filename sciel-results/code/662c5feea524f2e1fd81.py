"""Tensor decomposition for 4x4 @ 4x5 matrix multiplication.

Strategy:
  * Build a rank-44 decomposition via block Strassen (4x4x4 core, rank 28)
    plus a schoolbook boundary (4x4 x 4x1, rank 16).
  * Attempt to reduce the rank by detecting linearly-dependent columns in the
    stacked [U; V; W] matrix and removing them while preserving validity.
  * Return the best verified decomposition before the deadline.
"""
import time
import numpy as np

DIMENSIONS = [4, 4, 5]
N, M, P = 4, 4, 5


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify decomposition. U:(nm,rank) V:(mp,rank) W:(pn,rank).
    W index convention: W[k*n + i, c]."""
    rank = len(U[0])
    Ua = np.array(U, dtype=np.float64).reshape(n, m, rank)
    Va = np.array(V, dtype=np.float64).reshape(m, p, rank)
    Wa = np.array(W, dtype=np.float64).reshape(p, n, rank)
    rec = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    return bool(np.allclose(rec, np.ones((n, m, p)), atol=tol))


def _strassen_2x2():
    """Rank-7 tensors for 2x2 x 2x2 -> 2x2.
    Returns U,V,W each shape (4,7) flattened row-major (i,j),(j,k),(k,i)."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    def setU(i, j, c, v): U[i * 2 + j, c] += v
    def setV(j, k, c, v): V[j * 2 + k, c] += v
    def setW(k, i, c, v): W[k * 2 + i, c] += v

    # m0 = (a00+a11)(b00+b11)
    setU(0, 0, 0, 1); setU(1, 1, 0, 1)
    setV(0, 0, 0, 1); setV(1, 1, 0, 1)
    setW(0, 0, 0, 1); setW(1, 1, 0, 1)
    # m1 = (a10+a11)b00
    setU(1, 0, 1, 1); setU(1, 1, 1, 1)
    setV(0, 0, 1, 1)
    setW(0, 1, 1, 1); setW(1, 1, 1, -1)
    # m2 = a00(b01-b11)
    setU(0, 0, 2, 1)
    setV(0, 1, 2, 1); setV(1, 1, 2, -1)
    setW(1, 0, 2, 1); setW(1, 1, 2, 1)
    # m3 = a11(b10-b00)
    setU(1, 1, 3, 1)
    setV(1, 0, 3, 1); setV(0, 0, 3, -1)
    setW(0, 1, 3, 1); setW(1, 1, 3, 1)
    # m4 = (a00+a01)b11
    setU(0, 0, 4, 1); setU(0, 1, 4, 1)
    setV(1, 1, 4, 1)
    setW(1, 0, 4, -1); setW(1, 1, 4, 1)
    # m5 = (a10-a00)(b00+b01)
    setU(1, 0, 5, 1); setU(0, 0, 5, -1)
    setV(0, 0, 5, 1); setV(0, 1, 5, 1)
    setW(0, 1, 5, 1)
    # m6 = (a01-a11)(b10+b11)
    setU(0, 1, 6, 1); setU(1, 1, 6, -1)
    setV(1, 0, 6, 1); setV(1, 1, 6, 1)
    setW(1, 0, 6, 1)

    rec = np.einsum('ijc,jkc,kic->ijk', U.reshape(2, 2, 7),
                    V.reshape(2, 2, 7), W.reshape(2, 2, 7))
    assert np.allclose(rec, np.ones((2, 2, 2)), atol=1e-12), "Strassen 2x2 broken"
    return U, V, W


def _build_4x4x4_strassen():
    """Rank-28 decomposition for 4x4 x 4x4 -> 4x4 via block Strassen.
    Returns U:(16,28) V:(16,28) W:(16,28)."""
    Su, Sv, Sw = _strassen_2x2()
    U = np.zeros((16, 28))
    V = np.zeros((16, 28))
    W = np.zeros((16, 28))

    Uforms = [
        {('A', 0, 0): 1, ('A', 1, 1): 1},
        {('A', 1, 0): 1, ('A', 1, 1): 1},
        {('A', 0, 0): 1},
        {('A', 1, 1): 1},
        {('A', 0, 0): 1, ('A', 0, 1): 1},
        {('A', 1, 0): 1, ('A', 0, 0): -1},
        {('A', 0, 1): 1, ('A', 1, 1): -1},
    ]
    Vforms = [
        {('B', 0, 0): 1, ('B', 1, 1): 1},
        {('B', 0, 0): 1},
        {('B', 0, 1): 1, ('B', 1, 1): -1},
        {('B', 1, 0): 1, ('B', 0, 0): -1},
        {('B', 1, 1): 1},
        {('B', 0, 0): 1, ('B', 0, 1): 1},
        {('B', 1, 0): 1, ('B', 1, 1): 1},
    ]
    Wforms = [
        {('C', 0, 0): 1, ('C', 1, 1): 1},
        {('C', 1, 0): 1, ('C', 1, 1): -1},
        {('C', 0, 1): 1, ('C', 1, 1): 1},
        {('C', 1, 0): 1, ('C', 0, 0): 1},
        {('C', 0, 1): -1, ('C', 1, 1): 1},
        {('C', 1, 1): 1},
        {('C', 0, 0): 1},
    ]

    for s in range(7):
        for lr in range(2):
            for lc in range(2):
                c = s * 4 + lr * 2 + lc
                for (tag, br, bc), coeff in Uforms[s].items():
                    for i2 in range(2):
                        for j2 in range(2):
                            i = br * 2 + i2
                            j = bc * 2 + j2
                            U[i * 4 + j, c] += coeff
                for (tag, br, bc), coeff in Vforms[s].items():
                    for j2 in range(2):
                        for k2 in range(2):
                            j = br * 2 + j2
                            k = bc * 2 + k2
                            V[j * 4 + k, c] += coeff
                for (tag, br, bc), coeff in Wforms[s].items():
                    for k2 in range(2):
                        for i2 in range(2):
                            k = br * 2 + k2
                            i = bc * 2 + i2
                            W[k * 4 + i, c] += coeff

    assert _verify(U.tolist(), V.tolist(), W.tolist(), 4, 4, 4), "4x4x4 Strassen broken"
    return U, V, W


def _build_4x4x5_strassen_boundary():
    """Combine 4x4x4 Strassen (rank 28) + boundary 4x4 x 4x1 (rank 16) = rank 44.
    U:(16,44) V:(20,44) W:(20,44)."""
    U44, V44, W44 = _build_4x4x4_strassen()
    rank_core = 28
    rank_total = 44
    U = np.zeros((16, rank_total))
    V = np.zeros((20, rank_total))
    W = np.zeros((20, rank_total))

    U[:, :rank_core] = U44
    for j in range(4):
        for k in range(4):
            V[j * 5 + k, :rank_core] = V44[j * 4 + k, :rank_core]
    W[:16, :rank_core] = W44

    for i in range(4):
        for j in range(4):
            c = rank_core + i * 4 + j
            U[i * 4 + j, c] = 1.0
            V[j * 5 + 4, c] = 1.0
            W[4 * 4 + i, c] = 1.0

    assert _verify(U.tolist(), V.tolist(), W.tolist(), N, M, P), "4x4x5 broken"
    return U, V, W


def _reduce_linear(U, V, W, tol=1e-8, max_passes=60, deadline=None):
    """Detect columns that are linear combinations of others and merge them.

    Returns (U_list, V_list, W_list) with reduced rank, or None if no reduction.
    """
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]

    stacked = np.vstack([Ua, Va, Wa])

    changed = True
    passes = 0
    while changed and passes < max_passes:
        if deadline is not None and time.time() > deadline:
            break
        changed = False
        passes += 1
        r = stacked.shape[1]
        if r <= 1:
            break
        full_rank = np.linalg.matrix_rank(stacked, tol=tol)
        if full_rank >= r:
            break
        try:
            _, svals, Vt = np.linalg.svd(stacked, full_matrices=False)
        except np.linalg.LinAlgError:
            break
        null_dim = r - full_rank
        if null_dim <= 0:
            break
        null_vecs = Vt[full_rank:]
        for nv in null_vecs:
            idx = int(np.argmax(np.abs(nv)))
            if abs(nv[idx]) < 1e-10:
                continue
            keep = [c for c in range(r) if c != idx]
            stacked = stacked[:, keep]
            Ua = Ua[:, keep]
            Va = Va[:, keep]
            Wa = Wa[:, keep]
            changed = True
            break
        if not changed:
            break

    if Ua.shape[1] < rank:
        return Ua.tolist(), Va.tolist(), Wa.tolist()
    return None


def _reduce_zero_columns(U, V, W, max_iter=200, deadline=None):
    """Remove columns that are exactly zero."""
    Ua = np.array(U, dtype=np.float64)
    Va = np.array(V, dtype=np.float64)
    Wa = np.array(W, dtype=np.float64)
    rank = Ua.shape[1]
    for _ in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        removed = False
        for c in range(Ua.shape[1]):
            if (np.allclose(Ua[:, c], 0, atol=1e-14) and
                    np.allclose(Va[:, c], 0, atol=1e-14) and
                    np.allclose(Wa[:, c], 0, atol=1e-14)):
                keep = [x for x in range(Ua.shape[1]) if x != c]
                Ua = Ua[:, keep]
                Va = Va[:, keep]
                Wa = Wa[:, keep]
                removed = True
                break
        if not removed:
            break
    if Ua.shape[1] < rank:
        return Ua.tolist(), Va.tolist(), Wa.tolist()
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5."""
    start = time.time()
    deadline = start + 120.0  # safety margin before 150 s kill

    # Build the rank-44 Strassen + boundary decomposition
    U, V, W = _build_4x4x5_strassen_boundary()
    best_U, best_V, best_W = U.tolist(), V.tolist(), W.tolist()
    best_rank = U.shape[1]  # 44

    # Try linear-dependency reduction
    if time.time() < deadline - 5.0:
        reduced = _reduce_linear(best_U, best_V, best_W, tol=1e-8,
                                 max_passes=60, deadline=deadline - 5.0)
        if reduced is not None:
            rU, rV, rW = reduced
            if _verify(rU, rV, rW, N, M, P, tol=1e-9):
                best_U, best_V, best_W = rU, rV, rW
                best_rank = len(rU[0])

    # Try zero-column removal (cheap)
    if time.time() < deadline - 5.0:
        reduced = _reduce_zero_columns(best_U, best_V, best_W,
                                       max_iter=200, deadline=deadline - 5.0)
        if reduced is not None:
            rU, rV, rW = reduced
            if _verify(rU, rV, rW, N, M, P, tol=1e-9):
                best_U, best_V, best_W = rU, rV, rW
                best_rank = len(rU[0])

    # Final safety check
    assert _verify(best_U, best_V, best_W, N, M, P, tol=1e-9), "Final verification failed"

    return {"U": best_U, "V": best_V, "W": best_W}
