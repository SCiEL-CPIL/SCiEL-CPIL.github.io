"""Matrix multiplication 2x3x3 rank-minimal exact tensor decomposition search."""
import time
import numpy as np

_DIMS = (2, 3, 3)
_DEADLINE = 130.0
_START = time.time()


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed()


def _build_target():
    n, m, p = _DIMS
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _reconstruct(U, V, W):
    return np.einsum('ar,br,cr->abc', U, V, W)


def _als_step(U, V, W, T, r):
    n, m, p = _DIMS
    # Solve for U
    M_mat = np.einsum('br,cr->bcr', V, W).reshape(-1, r)
    T_mat = T.reshape(n * m, -1)
    U_new, _, _, _ = np.linalg.lstsq(M_mat.T, T_mat.T, rcond=None)
    U_new = U_new.T

    # Solve for V
    N_mat = np.einsum('ar,cr->acr', U, W).reshape(-1, r)
    T_mat_v = T.reshape(m * p, -1)
    V_new, _, _, _ = np.linalg.lstsq(N_mat.T, T_mat_v.T, rcond=None)
    V_new = V_new.T

    # Solve for W
    P_mat = np.einsum('ar,br->abr', U, V).reshape(-1, r)
    T_mat_w = T.reshape(p * n, -1)
    W_new, _, _, _ = np.linalg.lstsq(P_mat.T, T_mat_w.T, rcond=None)
    W_new = W_new.T

    return U_new, V_new, W_new


def _als(U, V, W, T, r, max_iter=300, tol=1e-14):
    for _ in range(max_iter):
        U_old, V_old, W_old = U.copy(), V.copy(), W.copy()
        U, V, W = _als_step(U, V, W, T, r)
        diff = (np.linalg.norm(U - U_old) +
                np.linalg.norm(V - V_old) +
                np.linalg.norm(W - W_old))
        if diff < tol:
            break
    return U, V, W


def _objective(U, V, W, T):
    T_approx = _reconstruct(U, V, W)
    return np.sum((T_approx - T) ** 2)


def _verify(U, V, W, T, tol=1e-7):
    T_approx = _reconstruct(U, V, W)
    if np.max(np.abs(T_approx - T)) > tol:
        return False
    for mat in (U, V, W):
        for val in mat.ravel():
            nearest = round(val * 2) / 2
            if abs(val - nearest) > 1e-5:
                return False
    return True


def _round_half(U, V, W):
    return np.round(U * 2) / 2, np.round(V * 2) / 2, np.round(W * 2) / 2


def _try_rank(rank, T, rng, n_restarts=80):
    n, m, p = _DIMS
    best = None
    best_obj = np.inf

    for _ in range(n_restarts):
        if _remaining() < 10:
            break

        scale = 0.5
        U = rng.standard_normal((n * m, rank)) * scale
        V = rng.standard_normal((m * p, rank)) * scale
        W = rng.standard_normal((p * n, rank)) * scale

        U, V, W = _als(U, V, W, T, rank, max_iter=300, tol=1e-14)
        obj = _objective(U, V, W, T)

        if obj < best_obj:
            best_obj = obj
            best = (U, V, W)

        if obj < 1e-10:
            break

    if best is None:
        return None

    U, V, W = best
    if _objective(U, V, W, T) > 1e-8:
        return None

    U_r, V_r, W_r = _round_half(U, V, W)
    if _verify(U_r, V_r, W_r, T, tol=1e-6):
        return U_r, V_r, W_r

    return None


def construct():
    n, m, p = _DIMS
    T = _build_target()
    rng = np.random.default_rng(42)

    rank_sb = n * m * p
    U0 = np.zeros((n * m, rank_sb))
    V0 = np.zeros((m * p, rank_sb))
    W0 = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j, col] = 1
                V0[j * p + k, col] = 1
                W0[k * n + i, col] = 1
                col += 1

    for rank in range(13, rank_sb):
        if _remaining() < 15:
            break
        result = _try_rank(rank, T, rng, n_restarts=80)
        if result is not None:
            U, V, W = result
            U_list = [[float(x) for x in row] for row in U]
            V_list = [[float(x) for x in row] for row in V]
            W_list = [[float(x) for x in row] for row in W]
            return {"U": U_list, "V": V_list, "W": W_list}

    U_list = [[int(x) for x in row] for row in U0]
    V_list = [[int(x) for x in row] for row in V0]
    W_list = [[int(x) for x in row] for row in W0]
    return {"U": U_list, "V": V_list, "W": W_list}
