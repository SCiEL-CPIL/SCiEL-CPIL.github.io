import numpy as np
import time

DIMENSIONS = [4, 4, 6]


def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _target(n, m, p):
    # Multiplication tensor: T[i*m+j, j*p+k, k*n+i] = 1
    E = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                E[i * m + j, j * p + k, k * n + i] = 1.0
    return E


def _als(E, rank, iters=3000, seed=0, tol=1e-10, deadline=None):
    rng = np.random.default_rng(seed)
    n1, n2, n3 = E.shape
    U = rng.normal(size=(n1, rank))
    V = rng.normal(size=(n2, rank))
    W = rng.normal(size=(n3, rank))
    E1 = E.reshape(n1, n2 * n3)
    E2 = E.reshape(n2, n1 * n3).T
    E3 = E.reshape(n3, n1 * n2).T
    best_err = np.inf
    best = (U.copy(), V.copy(), W.copy())
    for _ in range(iters):
        if deadline is not None and time.time() > deadline:
            break
        MVW = np.einsum('ar,br->ab', V, W).T
        U = np.linalg.lstsq(MVW.T, E1.T, rcond=None)[0].T
        MWV = np.einsum('ar,br->ab', W, V).T
        V = np.linalg.lstsq(MWV.T, E2.T, rcond=None)[0].T
        MUV = np.einsum('ar,br->ab', U, V).T
        W = np.linalg.lstsq(MUV.T, E3.T, rcond=None)[0].T
        recon = np.einsum('ar,br,cr->abc', U, V, W)
        err = float(np.linalg.norm(recon - E))
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy())
        if err < tol:
            break
    return best[0], best[1], best[2], best_err


def _is_exact(U, V, W, E, tol=1e-6):
    recon = np.einsum('ar,br,cr->abc', U, V, W)
    return float(np.linalg.norm(recon - E)) < tol


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115
    E = _target(n, m, p)
    rank = 24
    best = None
    best_err = np.inf
    for seed in range(8):
        if time.time() > deadline:
            break
        U, V, W, err = _als(E, rank, iters=3000, seed=seed, tol=1e-10, deadline=deadline)
        if err < best_err:
            best_err = err
            best = (U, V, W)
        if err < 1e-10:
            break
    if best is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best
        # snap to integer, then half-integer entries; keep the first exact one
        for scale in (1.0, 0.5):
            Ur = np.round(U / scale) * scale
            Vr = np.round(V / scale) * scale
            Wr = np.round(W / scale) * scale
            if _is_exact(Ur, Vr, Wr, E, tol=1e-6):
                U, V, W = Ur, Vr, Wr
                break
        if not _is_exact(U, V, W, E, tol=1e-6):
            U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
