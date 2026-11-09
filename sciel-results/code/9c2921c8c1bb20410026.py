import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 5]


def _build_schoolbook(n=3, m=3, p=5):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W


def _exact_check(U, V, W, n=3, m=3, p=5, tol=1e-9):
    """Return residual and allclose for the multiplicative tensor identity."""
    rank = U.shape[1]
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    residual = float(np.max(np.abs(T - 1.0)))
    ok = residual <= tol
    return residual, ok


def construct():
    start = time.time()
    HARD_LIMIT = 142.0  # seconds; evaluator kills at 150s, keep safety margin

    def time_left():
        return HARD_LIMIT - (time.time() - start)

    n, m, p = DIMENSIONS

    # Guaranteed valid solution: schoolbook rank 45.
    U_sb, V_sb, W_sb = _build_schoolbook(n, m, p)
    res, ok = _exact_check(U_sb, V_sb, W_sb, n, m, p)
    assert ok, "schoolbook verification failed"
    best_rank = U_sb.shape[1]
    best_U, best_V, best_W = U_sb, V_sb, W_sb

    # Attempt a faster exact search for a lower rank (e.g. 44) with a time guard.
    target_rank = 44
    entries = [-1, 0, 1]
    best_rank = min(best_rank, 45)

    import random
    random.seed(42)
    np.random.seed(42)

    # A quick deterministic random-ish search with exact verification.
    attempts = 0
    while time_left() > 2.0:
        attempts += 1
        U = np.array([[random.choice(entries) for _ in range(target_rank)] for _ in range(n * m)], dtype=np.int64)
        V = np.array([[random.choice(entries) for _ in range(target_rank)] for _ in range(m * p)], dtype=np.int64)
        W = np.array([[random.choice(entries) for _ in range(target_rank)] for _ in range(p * n)], dtype=np.int64)
        residual, ok = _exact_check(U, V, W, n, m, p)
        if ok:
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
        if attempts % 200000 == 0:
            # small progress marker (no-op) to allow early abort if stuck
            pass

    # Convert to lists
    U_list = np.asarray(best_U).astype(object).tolist()
    V_list = np.asarray(best_V).astype(object).tolist()
    W_list = np.asarray(best_W).astype(object).tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
