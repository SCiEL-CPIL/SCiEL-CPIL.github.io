"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _naive_decomp(n, m, p):
    """Build the naive rank = n*m*p decomposition."""
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    for j in range(m):
        for i in range(n):
            for k in range(p):
                c = i * m + j
                d = j * p + k
                e = k * n + i
                idx = j * (n * p) + i * p + k
                U[c, idx] = 1.0
                V[d, idx] = 1.0
                W[e, idx] = 1.0
    return U, V, W


def _drop_column_fast(U, V, W, col, keep_cols):
    """Try to drop column `col` by expressing W[:,col] as a combination of
    W[:,keep_cols]. Returns (new_U, new_V, new_W) or None."""
    Wc = W[:, col]
    Wk = W[:, keep_cols]
    try:
        a, res, rank, sv = np.linalg.lstsq(Wk, Wc, rcond=None)
    except Exception:
        return None
    if not np.allclose(Wk @ a, Wc, atol=1e-7, rtol=1e-7):
        return None
    new_U = U[:, keep_cols] + np.outer(U[:, col], a)
    new_V = V[:, keep_cols] + np.outer(V[:, col], a)
    new_W = W[:, keep_cols] + np.outer(W[:, col], a)
    return new_U, new_V, new_W


def _local_improve(U, V, W, time_budget):
    """Attempt to reduce rank via null-space column elimination."""
    start = time.time()
    cur_U, cur_V, cur_W = U.copy(), V.copy(), W.copy()
    cur_cols = list(range(cur_U.shape[1]))
    improved = True
    while improved:
        improved = False
        if time.time() - start > time_budget:
            break
        for col in list(cur_cols):
            if time.time() - start > time_budget:
                break
            keep_cols = [c for c in cur_cols if c != col]
            if not keep_cols:
                continue
            res = _drop_column_fast(cur_U, cur_V, cur_W, col, keep_cols)
            if res is not None:
                cur_U, cur_V, cur_W = res
                cur_cols.remove(col)
                improved = True
                break
    return cur_U, cur_V, cur_W, len(cur_cols)


def _to_int_matrices(U, V, W, cols):
    """Convert numpy float matrices to int lists."""
    U = np.rint(U).astype(int)
    V = np.rint(V).astype(int)
    W = np.rint(W).astype(int)
    return U.tolist(), V.tolist(), W.tolist()


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    DEADLINE = 135.0

    U, V, W = _naive_decomp(n, m, p)
    rank = U.shape[1]
    best = (rank, U.copy(), V.copy(), W.copy())

    if time.time() < DEADLINE:
        time_budget = max(1.0, DEADLINE - time.time() - 5.0)
        try:
            nU, nV, nW, new_rank = _local_improve(U, V, W, time_budget)
            if new_rank < best[0]:
                best = (new_rank, nU, nV, nW)
        except Exception:
            pass

    r, U, V, W = best
    U, V, W = _to_int_matrices(U, V, W, None)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
