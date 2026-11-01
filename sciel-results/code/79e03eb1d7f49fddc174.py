"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_result = schoolbook()
    best_rank = n * m * p

    def check(U, V, W, tol=1e-5):
        U = np.array(U)
        V = np.array(V)
        W = np.array(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    for target_rank in range(35, best_rank):
        if time.time() > deadline:
            break
        found = False
        for seed in range(3):
            if time.time() > deadline:
                break
            rng = np.random.RandomState(seed)
            r = target_rank
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            W = rng.randn(p * n, r)

            for it in range(300):
                if time.time() > deadline:
                    break
                for i in range(n):
                    for j in range(m):
                        M = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                for j in range(m):
                    for k in range(p):
                        M = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                        v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                for k in range(p):
                    for i in range(n):
                        M = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                        w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W[k * n + i, :] = w

            if check(U, V, W, tol=1e-4):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = target_rank
                found = True
                break
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
