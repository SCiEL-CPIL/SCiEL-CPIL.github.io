import time
import numpy as np

DIMENSIONS = [2, 2, 10]
_DEADLINE = 120.0

def _build_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p, T):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    recon = np.einsum('ir,jr,kr->ijk', U, V, W)
    return bool(np.allclose(recon, T, atol=1e-9, rtol=0.0))

_STRASSEN = [
    ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
    ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
    ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
    ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
    ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]),
    ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
    ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
]

def _emit_strassen(U, V, W, n, p, base_col, c0, c1):
    for t in range(7):
        col = base_col + t
        u_side, v_side, w_side = _STRASSEN[t]
        for u_row, coeff in u_side:
            U[u_row][col] += coeff
        for v_koff, coeff in v_side:
            j = v_koff // 2
            k = c0 if v_koff % 2 == 0 else c1
            V[j * p + k][col] += coeff
        for w_koff, coeff in w_side:
            i = w_koff // 2
            k = c0 if w_koff % 2 == 0 else c1
            W[k * n + i][col] += coeff

def _strassen_rank35(n, m, p):
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for b in range(5):
        _emit_strassen(U, V, W, n, p, b * 7, 2 * b, 2 * b + 1)
    return U, V, W

def _round_half(X):
    return np.round(X * 2) / 2

def _als(U0, V0, W0, T, n, m, p, R, iters=100, polish_iters=60, deadline=None, t0=None):
    U = U0.astype(float).copy()
    V = V0.astype(float).copy()
    W = W0.astype(float).copy()
    I, J, K = n * m, m * p, p * n

    def _upd_U():
        M = np.einsum('jr,kr->jkr', V, W).reshape(J * K, R)
        U[:] = np.linalg.lstsq(M, T.reshape(I, J * K).T, rcond=None)[0].T

    def _upd_V():
        M = np.einsum('ir,kr->ikr', U, W).reshape(I * K, R)
        V[:] = np.linalg.lstsq(M, T.reshape(J, I * K).T, rcond=None)[0].T

    def _upd_W():
        M = np.einsum('ir,jr->ijr', U, V).reshape(I * J, R)
        W[:] = np.linalg.lstsq(M, T.reshape(K, I * J).T, rcond=None)[0].T

    def _check():
        Ur, Vr, Wr = _round_half(U), _round_half(V), _round_half(W)
        recon = np.einsum('ir,jr,kr->ijk', Ur, Vr, Wr)
        if np.linalg.norm(recon - T) < 1e-9:
            return Ur, Vr, Wr
        return None

    def _timeup():
        return deadline is not None and (time.time() - t0) > deadline

    for _ in range(iters):
        if _timeup():
            break
        _upd_U(); _upd_V(); _upd_W()
        res = _check()
        if res is not None:
            return res
    for _ in range(polish_iters):
        if _timeup():
            break
        _upd_U(); _upd_V(); _upd_W()
    return _check()

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    U35, V35, W35 = _strassen_rank35(n, m, p)
    if not _verify(U35, V35, W35, n, m, p, T):
        raise RuntimeError("rank-35 baseline failed verification")
    best = (35, U35, V35, W35)

    def _remaining():
        return _DEADLINE - (time.time() - t0) - 10.0

    U35f = np.array(U35, dtype=float)
    V35f = np.array(V35, dtype=float)
    W35f = np.array(W35, dtype=float)

    for R in [34, 33, 32]:
        if _remaining() < 6.0:
            break
        found = None
        if R == 34:
            for drop in range(5):
                if _remaining() < 4.0:
                    break
                keep = [c for c in range(35) if c != drop]
                res = _als(U35f[:, keep], V35f[:, keep], W35f[:, keep],
                           T, n, m, p, R, iters=60, polish_iters=30,
                           deadline=_DEADLINE - 10.0, t0=t0)
                if res is not None:
                    Ur, Vr, Wr = res
                    if _verify(Ur, Vr, Wr, n, m, p, T):
                        found = (Ur, Vr, Wr)
                        break
        else:
            rng = np.random.default_rng(R)
            for seed in range(3):
                if _remaining() < 4.0:
                    break
                U0 = rng.standard_normal((n * m, R))
                V0 = rng.standard_normal((m * p, R))
                W0 = rng.standard_normal((p * n, R))
                res = _als(U0, V0, W0, T, n, m, p, R, iters=60, polish_iters=30,
                           deadline=_DEADLINE - 10.0, t0=t0)
                if res is not None:
                    Ur, Vr, Wr = res
                    if _verify(Ur, Vr, Wr, n, m, p, T):
                        found = (Ur, Vr, Wr)
                        break
        if found is not None:
            Ur, Vr, Wr = found
            best = (R, Ur.tolist(), Vr.tolist(), Wr.tolist())

    R, U, V, W = best
    return {"U": U, "V": V, "W": W}
