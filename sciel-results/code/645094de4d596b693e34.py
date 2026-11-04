import time
import math
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def _to_int_list(A):
    A = np.asarray(A, dtype=float)
    return [[float(round(float(v))) for v in row] for row in A.tolist()]

def construct():
    t0 = time.time()
    DEADLINE = t0 + 120.0          # safety margin vs 150s kill
    n, m, p = DIMENSIONS
    R0 = n * m * p                  # schoolbook rank = 42

    def build_schoolbook():
        U = np.zeros((n * m, R0)); V = np.zeros((m * p, R0)); W = np.zeros((p * n, R0))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1.0
                    V[j * p + k, c] = 1.0
                    W[k * n + i, c] = 1.0
                    c += 1
        return U, V, W

    def residual(U, V, W):
        # F = sum_c U_c V_c^T  (n*m x m*p); G = sum_c W_c^T U_c  (m*p x n*m)
        F = (U * V).sum(axis=1)
        G = (W.T * U).sum(axis=1)
        return float(np.linalg.norm(F - G.T))

    def valid(U, V, W):
        return residual(U, V, W) < 1e-6

    def rank_of(U, V, W):
        nz = ((np.abs(U) > 1e-9) | (np.abs(V) > 1e-9) | (np.abs(W) > 1e-9)).any(axis=0)
        return int(nz.sum())

    def round3(A):
        return np.round(A, 3)

    # --- baseline ---
    U0, V0, W0 = build_schoolbook()
    best = (residual(U0, V0, W0), U0.copy(), V0.copy(), W0.copy())
    best_valid = (R0, U0.copy(), V0.copy(), W0.copy())

    # --- local search with dead-column elimination ---
    def search(U, V, W, iters, lr, seed):
        rng = random.Random(seed)
        U = U.copy(); V = V.copy(); W = W.copy()
        for it in range(iters):
            if time.time() > DEADLINE - 8.0:
                break
            # gradient step
            F = (U * V).sum(axis=1)
            G = (W.T * U).sum(axis=1)
            dU = (F - G.T) * V
            dV = (F - G.T).T * U
            dW = (G - F.T).T * U
            U = U - lr * dU
            V = V - lr * dV
            W = W - lr * dW
            # occasional small perturbation to escape
            if it % 25 == 0:
                s = 0.02
                U = U + s * rng.gauss(0, 1) * (np.abs(U) + 1)
                V = V + s * rng.gauss(0, 1) * (np.abs(V) + 1)
                W = W + s * rng.gauss(0, 1) * (np.abs(W) + 1)
            # dead elimination
            nz = ((np.abs(U) > 1e-5) | (np.abs(V) > 1e-5) | (np.abs(W) > 1e-5))
            if not nz.all():
                U = U[:, nz]; V = V[:, nz]; W = W[:, nz]
            r = residual(U, V, W)
            if r < best[0]:
                best = (r, U.copy(), V.copy(), W.copy())
            if r < 1e-6:
                rk = rank_of(U, V, W)
                if rk < best_valid[0]:
                    best_valid = (rk, U.copy(), V.copy(), W.copy())
        return U, V, W

    # a few restarts from the schoolbook start with different lrs
    for seed, lr in [(1, 0.02), (2, 0.05), (3, 0.01)]:
        if time.time() > DEADLINE - 10.0:
            break
        U, V, W = search(U0, V0, W0, iters=400, lr=lr, seed=seed)

    # pick the best valid if any, else best residual (still returns a decomposition)
    if best_valid[0] < R0:
        rk, U, V, W = best_valid
    else:
        _, U, V, W = best
    U, V, W = round3(U), round3(V), round3(W)

    return {"U": _to_int_list(U), "V": _to_int_list(V), "W": _to_int_list(W)}
# EVOLVE-BLOCK-END
