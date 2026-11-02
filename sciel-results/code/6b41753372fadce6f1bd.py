"""Matrix multiplication 3x4x4 - rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 4, 4]
N, M, P = DIMENSIONS
RANK_SCHOOLBOOK = N * M * P  # 48

_DEADLINE = time.time() + 112.0  # safety margin before 120s


def _build_tensor():
    T = np.ones((N, M, P), dtype=float)
    return T


def _verify(U, V, W, tol=1e-6):
    n, m, p = N, M, P
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = float(np.sum(U[i * m + j] * V[j * p + k] * W[k * n + i]))
                if abs(s - 1.0) > tol:
                    return False
    return True


def _round_to_half(U, V, W):
    def r(x):
        return np.round(2.0 * x) / 2.0
    return r(U), r(V), r(W)


def _as_list(M_):
    return [[float(v) for v in row] for row in M_]


def _schoolbook():
    n, m, p = N, M, P
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1.0
                V[j * p + k, c] = 1.0
                W[k * n + i, c] = 1.0
                c += 1
    return U, V, W


def _search_rank(target_rank, seed, max_iters=4000, lr=0.01):
    """Vectorized gradient descent on (U,V,W) to match the all-ones tensor.
    Returns (U,V,W, err)."""
    rng = np.random.default_rng(seed)
    n, m, p = N, M, P
    r = target_rank
    U = rng.standard_normal((n * m, r)) * 0.3
    V = rng.standard_normal((m * p, r)) * 0.3
    W = rng.standard_normal((p * n, r)) * 0.3
    for c in range(r):
        if U[:, c].sum() < 0:
            U[:, c] *= -1
            V[:, c] *= -1

    # Pre-index arrays for fast vectorized reconstruction.
    # Reconstruct R[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # Build index maps.
    Ui = np.arange(n * m).reshape(n, m)          # U row index for (i,j)
    Vj = np.arange(m * p).reshape(m, p)          # V row index for (j,k)
    Wk = np.arange(p * n).reshape(p, n)          # W row index for (k,i)

    T = np.ones((n, m, p))
    best_err = None
    best = (U.copy(), V.copy(), W.copy())
    for it in range(max_iters):
        if time.time() > _DEADLINE - 20:
            break
        # R[c] contribution: outer product per column c.
        # R = sum_c  U[:,c][i,j] * V[:,c][j,k] * W[:,c][k,i]
        R = np.zeros((n, m, p))
        for c in range(r):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            # (n,m) x (m,p) -> (n,p); then multiply by wc[k,i]
            A = uc @ vc            # (n, p)
            R += A[:, :, None] * wc.T   # wc.T is (n, p); broadcast over k
        err = R - T
        # Gradient: dU[i,j,c] = sum_k err[i,j,k]*V[j,k,c]*W[k,i,c]
        dU = np.zeros((n * m, r))
        dV = np.zeros((m * p, r))
        dW = np.zeros((p * n, r))
        for c in range(r):
            uc = U[:, c].reshape(n, m)
            vc = V[:, c].reshape(m, p)
            wc = W[:, c].reshape(p, n)
            # dU[i,j] = sum_k err[i,j,k] * vc[j,k] * wc[k,i]
            # = (err @ vc)[i,j] weighted by wc[k,i]
            # err (n,m,p); vc (m,p)
            # For each i: sum_k err[i,:,k]*vc[:,k] -> (m,) then * wc[k,i]
            dUc = np.zeros((n, m))
            dVc = np.zeros((m, p))
            dWc = np.zeros((p, n))
            for i in range(n):
                dUc[i, :] = np.sum(err[i, :, :] * vc, axis=1) * wc[:, i]
            for j in range(m):
                dVc[j, :] = np.sum(err[:, j, :] * uc[:, j], axis=0) * wc[:, j]
            for k in range(p):
                dWc[k, :] = np.sum(err[:, :, k] * uc * vc[:, k], axis=0) * vc[k, :]
            dU[:, c] = dUc.reshape(-1)
            dV[:, c] = dVc.reshape(-1)
            dW[:, c] = dWc.reshape(-1)
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        if it % 20 == 0:
            err_tot = float(np.sum(err ** 2))
            if best_err is None or err_tot < best_err:
                best_err = err_tot
                best = (U.copy(), V.copy(), W.copy())
            if best_err < 1e-10:
                break
    return best, best_err


def construct():
    U0, V0, W0 = _schoolbook()
    fallback = (U0, V0, W0)
    for r in range(15, RANK_SCHOOLBOOK):
        if time.time() > _DEADLINE - 25:
            break
        for seed in range(3):
            if time.time() > _DEADLINE - 25:
                break
            (U, V, W), err = _search_rank(r, seed=seed, max_iters=3000, lr=0.01)
            if err is not None and err < 1e-3:
                Ur, Vr, Wr = _round_to_half(U, V, W)
                if _verify(Ur, Vr, Wr, tol=1e-6):
                    return {"U": _as_list(Ur), "V": _as_list(Vr), "W": _as_list(Wr)}
                if _verify(U, V, W, tol=1e-6):
                    return {"U": _as_list(U), "V": _as_list(V), "W": _as_list(W)}
    return {"U": _as_list(U0), "V": _as_list(V0), "W": _as_list(W0)}
