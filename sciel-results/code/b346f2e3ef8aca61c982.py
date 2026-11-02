"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]
import time
import math
import random
import numpy as np

N, M, P = 2, 3, 7
S = N * M * P  # 42 (schoolbook rank / seed)


def _idx_u(i, j, k):
    return i * M + j


def _idx_v(i, j, k):
    return j * P + k


def _idx_w(i, j, k):
    return k * N + i


def _schoolbook():
    n, m, p = N, M, P
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}


def _matmul_check(U, V, W, R):
    # U: n*m x R, V: m*p x R, W: p*n x R
    # C_{i,k} = sum_j A_{i,j} B_{j,k};  C_{i,k} = sum_r U_{i,m?} ... verify
    Ua = np.asarray(U, dtype=float)
    Va = np.asarray(V, dtype=float)
    Wa = np.asarray(W, dtype=float)
    # Build tensors
    n, m, p = N, M, P
    # U_{i,j,r}, V_{j,k,r}, W_{k,i,r}
    U3 = Ua.reshape(n, m, R)
    V3 = Va.reshape(m, p, R)
    W3 = Wa.reshape(p, n, R)
    # For each r, rank-1 contribution to T: U3[:, :, r][i,j] * V3[:, :, r][j,k] * W3[:, :, r][k,i]
    # Sum over r of U[i,j,r]*V[j,k,r]*W[k,i,r] should equal delta(i,j,k) = 1 if consistent
    # T_{i,j,k} = 1 for all i,j,k (the multiplication tensor entries are 1)
    T = np.zeros((n, m, p))
    for r in range(R):
        T += U3[:, :, r][:, :, None] * V3[:, :, r][None, :, :] * W3[:, :, r][None, None, :]
    # target: 1 everywhere
    if np.allclose(T, np.ones((n, m, p)), atol=1e-6, rtol=1e-6):
        return True
    return False


def _check_half_int(U, V, W):
    for Mx in (U, V, W):
        for row in Mx:
            for x in row:
                # must be integer or half-integer
                if abs(x * 2 - round(x * 2)) > 1e-9:
                    return False
    return True


def _snap_half(U, V, W):
    def snp(Mx):
        return [[round(x * 2) / 2.0 for x in row] for row in Mx]
    return snp(U), snp(V), snp(W)


def _greedy_peel(seed, target_rank, time_budget):
    """Greedy rank-1 peeling on dense float init, snap to half-int, verify."""
    rng = np.random.default_rng(seed)
    n, m, p = N, M, P
    # target tensor = ones
    T = np.ones((n, m, p))
    remaining = T.copy()
    colsU, colsV, colsW = [], [], []
    t0 = time.time()
    for _ in range(target_rank):
        if time.time() - t0 > time_budget:
            break
        # best rank-1 approx of remaining: maximize |<remaining, u v w>|
        # iterate ALS a few times from random init
        best_score = 0
        best_uvw = None
        for trial in range(30):
            u = rng.normal(size=(n, m))
            v = rng.normal(size=(m, p))
            w = rng.normal(size=(p, n))
            for _ in range(8):
                u = np.tensordot(remaining, np.tensordot(v, w, axes=([1], [0])), axes=([1, 2], [1, 0]))
                v = np.tensordot(remaining, np.tensordot(w, u, axes=([0], [0])), axes=([2, 0], [1, 0]))
                w = np.tensordot(remaining, np.tensordot(u, v, axes=([1], [1])), axes=([0, 2], [0, 1]))
                u = u / (np.linalg.norm(u) + 1e-15)
                v = v / (np.linalg.norm(v) + 1e-15)
                w = w / (np.linalg.norm(w) + 1e-15)
            score = np.tensordot(remaining, np.tensordot(np.tensordot(u, v, axes=([1], [1])), w, axes=([1], [0])), axes=([0, 1, 2], [0, 1, 2]))
            if score > best_score:
                best_score = score
                best_uvw = (u, v, w)
        if best_score <= 1e-9:
            break
        u, v, w = best_uvw
        # normalize to make contribution exact: scale so that it matches sign
        # add this rank-1
        colsU.append(u.ravel())
        colsV.append(v.ravel())
        colsW.append(w.ravel())
        remaining -= np.tensordot(np.tensordot(u, v, axes=([1], [1])), w, axes=([1], [0]))
    # assemble
    R = len(colsU)
    if R == 0:
        return None
    Uf = np.column_stack(colsU).reshape(n * m, R)
    Vf = np.column_stack(colsV).reshape(m * p, R)
    Wf = np.column_stack(colsW).reshape(p * n, R)
    # snap to half-int
    U = [[round(x * 2) / 2.0 for x in row] for row in Uf.tolist()]
    V = [[round(x * 2) / 2.0 for x in row] for row in Vf.tolist()]
    W = [[round(x * 2) / 2.0 for x in row] for row in Wf.tolist()]
    if _matmul_check(U, V, W, R):
        return {"U": U, "V": V, "W": W}
    return None


# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 115.0  # safety margin before 120s
    # try to find a lower-rank exact decomposition
    best = None
    # greedy peeling with several seeds, target ranks descending
    for target_rank in (34, 30, 26, 22):
        for seed in (0, 1, 2, 3):
            if time.time() - t0 > deadline:
                break
            cand = _greedy_peel(seed, target_rank, time_budget=deadline - (time.time() - t0))
            if cand is not None:
                best = cand
                break
        if best is not None:
            break
    if best is None:
        best = _schoolbook()
    return best
# EVOLVE-BLOCK-END
