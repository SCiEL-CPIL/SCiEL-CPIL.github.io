"""Search for a low-rank exact tensor decomposition of 4x4 by 4x5 matmul."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]
DEADLINE = 150.0


def _remaining_budget():
    return DEADLINE - (time.time() - _START)


_START = time.time()


def _build_tensor(n, m, p):
    """Return flat tensor T[i, j, k] = 1 if (i,j,k) is a valid term, else 0."""
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1
    return T


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check sum_r U[i*m+j, r] * V[j*p+k, r] == 1 for all (i,j,k)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    rank = U.shape[1]
    ok = True
    for i in range(n):
        for j in range(m):
            # contribution of term (i,j) across k
            rowU = U[i * m + j, :]          # (rank,)
            for k in range(p):
                rowV = V[j * p + k, :]      # (rank,)
                s = float(np.dot(rowU, rowV))
                if abs(s - 1.0) > tol:
                    ok = False
                    return False
    return ok


def _try_peel(T, n, m, p, max_rank, rng):
    """Greedy rank-1 peeling with verification; returns (rank, U, V, W) or None."""
    n, m, p = int(n), int(m), int(p)
    T = np.array(T, dtype=float)
    U = []
    V = []
    W = []
    work = T.copy()
    for r in range(1, max_rank + 1):
        if _remaining_budget() < 2.0:
            return None
        # find best (i,j,k) with max |work|
        idx = np.unravel_index(np.argmax(np.abs(work)), work.shape)
        val = work[idx]
        if abs(val) < 1e-12:
            return None
        i, j, k = idx
        # rank-1 factor: u_i=1, v_j=1, w_k=1/val
        u = np.zeros(n * m)
        v = np.zeros(m * p)
        w = np.zeros(p * n)
        u[i * m + j] = 1.0
        v[j * p + k] = 1.0
        w[k * n + i] = 1.0 / val
        # subtract
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    work[ii, jj, kk] -= u[ii * m + jj] * v[jj * p + kk] * w[kk * n + ii]
        U.append(u)
        V.append(v)
        W.append(w)
        if _remaining_budget() < 2.0:
            return None
        if np.max(np.abs(work)) < 1e-12:
            break
    else:
        # exhausted max_rank without zero
        if np.max(np.abs(work)) > 1e-12:
            return None
    U = np.array(U).T
    V = np.array(V).T
    W = np.array(W).T
    if not _verify(U, V, W, n, m, p):
        return None
    return (U.shape[1], U, V, W)


def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)

    # Schoolbook fallback (always valid, rank 80)
    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, c] = 1
                V_fb[j * p + k, c] = 1
                W_fb[k * n + i, c] = 1
                c += 1

    best = (rank_fb, U_fb, V_fb, W_fb)

    # Try greedy peeling from a few seeds / starting indices
    seeds = [0, 1, 2, 3, 12345]
    for seed in seeds:
        if _remaining_budget() < 5.0:
            break
        rng = random.Random(seed)
        res = _try_peel(T, n, m, p, 60, rng)
        if res is not None:
            rk, U, V, W = res
            if rk < best[0]:
                best = (rk, U, V, W)

    rk, U, V, W = best
    return {"U": U, "V": V, "W": W}
