"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _residuals(U, V, W):
    n, m, p = DIMENSIONS
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return T - 1.0

def _verify(U, V, W, tol=1e-9):
    for M in (U, V, W):
        scaled = 2.0 * M
        if not np.allclose(scaled, np.round(scaled), atol=1e-6):
            return False
    r = _residuals(U, V, W)
    return np.max(np.abs(r)) < tol

def _round_half_int(U, V, W):
    return (np.round(2.0 * U) / 2.0, np.round(2.0 * V) / 2.0, np.round(2.0 * W) / 2.0)

def _als_once(r, deadline, rng, n, m, p):
    sizeU, sizeV, sizeW = n * m, m * p, p * n
    U = rng.standard_normal((sizeU, r)) * 0.5
    V = rng.standard_normal((sizeV, r)) * 0.5
    W = rng.standard_normal((sizeW, r)) * 0.5
    b = np.ones(n * m * p)
    for it in range(150):
        if time.time() > deadline:
            return None
        # Solve for U given V, W
        M = np.zeros((n * m * p, sizeU * r))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[idx, (i * m + j) * r:(i * m + j + 1) * r] = V[j * p + k] * W[k * n + i]
                    idx += 1
        sol, *_ = np.linalg.lstsq(M, b, rcond=None)
        U = sol.reshape(sizeU, r)
        # Solve for V given U, W
        M = np.zeros((n * m * p, sizeV * r))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[idx, (j * p + k) * r:(j * p + k + 1) * r] = U[i * m + j] * W[k * n + i]
                    idx += 1
        sol, *_ = np.linalg.lstsq(M, b, rcond=None)
        V = sol.reshape(sizeV, r)
        # Solve for W given U, V
        M = np.zeros((n * m * p, sizeW * r))
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    M[idx, (k * n + i) * r:(k * n + i + 1) * r] = U[i * m + j] * V[j * p + k]
                    idx += 1
        sol, *_ = np.linalg.lstsq(M, b, rcond=None)
        W = sol.reshape(sizeW, r)
        R = _residuals(U, V, W)
        if np.max(np.abs(R)) < 1e-7:
            Ur, Vr, Wr = _round_half_int(U, V, W)
            if _verify(Ur, Vr, Wr):
                return Ur, Vr, Wr
    Ur, Vr, Wr = _round_half_int(U, V, W)
    if _verify(Ur, Vr, Wr):
        return Ur, Vr, Wr
    return None

def _try_rank(r, deadline, rng):
    n, m, p = DIMENSIONS
    for _ in range(30):
        if time.time() > deadline:
            return None
        res = _als_once(r, deadline, rng, n, m, p)
        if res is not None:
            return res
    return None

def construct():
    deadline = time.time() + 130.0
    rng = random.Random(12345)
    U0, V0, W0 = _schoolbook()
    best = (U0, V0, W0)
    best_rank = U0.shape[1]
    candidate_ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64, 72, 80, 90, 100]
    for r in candidate_ranks:
        if time.time() > deadline - 8.0:
            break
        if r >= best_rank:
            break
        res = _try_rank(r, deadline, rng)
        if res is not None:
            U, V, W = res
            if U.shape[1] < best_rank:
                best = (U, V, W)
                best_rank = U.shape[1]
    U, V, W = best
    def clean(x):
        r = round(x * 2) / 2.0
        return 0.0 if r == 0 else r
    U_list = [[clean(v) for v in row] for row in U.tolist()]
    V_list = [[clean(v) for v in row] for row in V.tolist()]
    W_list = [[clean(v) for v in row] for row in W.tolist()]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
