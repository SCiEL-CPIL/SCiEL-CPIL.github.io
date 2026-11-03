"""GRASP + vectorized 2-swap local search for the QAPLIB tho150 instance."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _grasp_construct(a, b, C, rng, alpha):
    """Construct a solution by greedily assigning facilities to locations,
    choosing each time from a randomized restricted candidate list."""
    n = a.shape[0]
    p = np.empty(n, dtype=np.int64)
    facs = np.arange(n)
    locs = np.arange(n)
    # R[loc] = sum over already-placed facilities f of b[loc, p[f]] * a[f, :]
    R = np.zeros((n, n), dtype=np.float64)
    for _ in range(n):
        # immediate cost of placing facility f at location l:
        #   a[f,f]*b[l,l] + 2 * ( b[l] @ a[:, f] + R[l] @ a[f, :] )
        # vectorize over (f, l)
        imm = np.zeros((len(facs), len(locs)), dtype=np.float64)
        for i, f in enumerate(facs):
            imm[i] = (
                a[f, f] * b[locs, locs]
                + 2.0 * (b[locs] @ a[:, f] + R[locs] @ a[f, :])
            )
        minv = imm.min(axis=0)
        maxv = imm.max(axis=0)
        span = np.where(maxv - minv < 1e-9, 1.0, maxv - minv)
        thresh = minv + alpha * span
        ok = imm <= thresh
        col_has = ok.any(axis=0)
        cand_cols = np.where(col_has)[0]
        c = cand_cols[rng.integers(0, len(cand_cols))]
        cand_rows = np.where(ok[:, c])[0]
        r = cand_rows[rng.integers(0, len(cand_rows))]
        f = facs[r]
        l = locs[c]
        p[f] = l
        # update R: for each location loc, R[loc] += b[loc, l] * a[f, :]
        R += b[:, l][:, None] * a[f, :][None, :]
        facs = np.delete(facs, r)
        locs = np.delete(locs, c)
    return p


def _recompute_R(a, b, p):
    n = a.shape[0]
    idx = np.arange(n)
    # R[loc] = sum_f b[loc, p[f]] * a[f, :]
    R = np.zeros((n, n), dtype=np.float64)
    for loc in range(n):
        R[loc] = (b[loc, p] * a[idx, :]).sum(axis=0)
    return R


def _local_search(a, b, C, p, deadline):
    n = a.shape[0]
    idx = np.arange(n)
    R = _recompute_R(a, b, p)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        li = p[:, None]
        lj = p[None, :]
        ci = C[idx[:, None], idx[:, None]]
        cj = C[idx[None, :], idx[None, :]]
        cij = C[idx[:, None], idx[None, :]]
        Rli = R[li]
        Rlj = R[lj]
        bli_li = b[li, li]
        blj_lj = b[lj, lj]
        bli_lj = b[li, lj]
        delta = (
            -cij * (Rli + Rlj - ci - cj)
            + (ci + cj) * (bli_li + blj_lj)
            - 2.0 * (ci + cj - cij) * bli_lj
        )
        np.fill_diagonal(delta, 0.0)
        iu, ju = np.triu_indices(n, k=1)
        d = delta[iu, ju]
        k = int(np.argmin(d))
        if d[k] >= -1e-7:
            break
        i, j = iu[k], ju[k]
        p[i], p[j] = p[j], p[i]
        li = p[i]
        lj = p[j]
        R[li] = (b[li, p] * a[idx, :]).sum(axis=0)
        R[lj] = (b[lj, p] * a[idx, :]).sum(axis=0)
        improved = True
    return p


def solve(a, b):
    start = time.time()
    deadline = start + 140.0
    n = a.shape[0]
    rng = np.random.default_rng(12345)
    C = a + a.T
    idx = np.arange(n)

    def cost(p):
        return int((a[:, p] * b[idx[:, None], p]).sum())

    # Baseline greedy start
    p0 = _grasp_construct(a, b, C, rng, alpha=0.0)
    best_p = p0.copy()
    best_cost = cost(p0)

    # GRASP loop with local search
    while time.time() < deadline:
        alpha = 0.2 + 0.5 * rng.random()
        p = _grasp_construct(a, b, C, rng, alpha)
        c = cost(p)
        if c < best_cost:
            best_p = p.copy()
            best_cost = c
        # local search
        p = _local_search(a, b, C, p, deadline)
        c = cost(p)
        if c < best_cost:
            best_p = p.copy()
            best_cost = c

    if best_p is None:
        best_p = idx.copy()
    return best_p.tolist()
# EVOLVE-BLOCK-END
