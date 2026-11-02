"""Rank-minimal exact tensor decomposition for 3x5 * 5x6 matrix multiplication."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 6]
START = time.time()
DEADLINE = START + 138.0  # safety margin before 150s kill
TOL = 1e-7


def _schoolbook(n, m, p):
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


def _tensor_residual(U, V, W, n, m, p):
    """Return max |T - sum_c U_c V_c W_c| over the (n*m*p) tensor entries."""
    nmp = n * m * p
    Uf = np.asarray(U, dtype=float)
    Vf = np.asarray(V, dtype=float)
    Wf = np.asarray(W, dtype=float)
    r = Uf.shape[1]
    # Build full T via einsum: T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    U3 = Uf.reshape(n, m, r)
    V3 = Vf.reshape(m, p, r)
    W3 = Wf.reshape(p, n, r)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3, optimize=True)
    # target: T[i,j,k] = 1 iff k==i (independent of j)
    i = np.arange(n)
    target = (i[:, None, None] == np.arange(p)[None, None, :])
    # target shape (n, p); broadcast over m
    target = np.broadcast_to(target[:, None, :], (n, m, p)).astype(float)
    return float(np.max(np.abs(T - target)))


def _validate(U, V, W, n, m, p):
    if _tensor_residual(U, V, W, n, m, p) > TOL:
        return False
    return True


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook(n, m, p)
    best = (U.copy(), V.copy(), W.copy())
    best_rank = U.shape[1]
    rng = random.Random(12345)

    def try_reduce(U, V, W):
        nonlocal best, best_rank
        r = U.shape[1]
        if r <= 1:
            return U, V, W
        # attempt to reduce rank by 1: combine two random columns
        for _ in range(40):
            if time.time() > DEADLINE:
                return U, V, W
            a, b = rng.sample(range(r), 2)
            alpha = rng.choice([0.5, -0.5, 0.25, -0.25, 1.0, -1.0])
            U2 = U.copy()
            V2 = V.copy()
            W2 = W.copy()
            U2[:, a] = U2[:, a] + alpha * U2[:, b]
            V2[:, a] = V2[:, a] + alpha * V2[:, b]
            W2[:, a] = W2[:, a] - alpha * W2[:, b]
            # drop column b
            U3 = np.delete(U2, b, axis=1)
            V3 = np.delete(V2, b, axis=1)
            W3 = np.delete(W2, b, axis=1)
            if _validate(U3, V3, W3, n, m, p):
                r = r - 1
                U, V, W = U3, V3, W3
                if r < best_rank:
                    best_rank = r
                    best = (U3.copy(), V3.copy(), W3.copy())
                break
        return U, V, W

    # Main loop: keep reducing rank until stuck, then try random restarts
    guard = 0
    while time.time() < DEADLINE and U.shape[1] > 1:
        before = U.shape[1]
        U, V, W = try_reduce(U, V, W)
        guard += 1
        if U.shape[1] == before:
            # stuck; perturb by random linear transform of columns to escape local min
            if guard % 5 == 0:
                r = U.shape[1]
                M = np.eye(r)
                for _ in range(3):
                    a, b = rng.sample(range(r), 2)
                    alpha = rng.choice([0.5, -0.5, 0.25, -0.25])
                    M[:, a] = M[:, a] + alpha * M[:, b]
                U = U @ M
                V = V @ M
                W = W @ M
    # final: ensure we return best found
    U, V, W = best
    return {"U": U, "V": V, "W": W}
