# EVOLVE-BLOCK-START
import numpy as np
import time


def _qap_cost(a, b, p):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _compute_C(a, b, p):
    # C[i] = sum_j a[i,j] * B[p[i], p[j]]  with B = b[p,p]
    B = b[np.ix_(p, p)]
    return (a @ B).astype(np.int64)


def _refine(a, b, p, C, deadline, best_cost, best_p):
    """Best-improvement 2-opt (swap) local search until deadline.
    p is a numpy int64 array (a copy); C is updated in place.
    Returns (best_cost, best_p) found so far."""
    n = len(p)
    # current cost = sum(C)
    cur = int(C.sum())
    while time.time() < deadline:
        improved = False
        best_delta = 0
        bi = -1
        bj = -1
        for i in range(n):
            ai = a[i]
            pi = p[i]
            for j in range(i + 1, n):
                d = int(ai[j] * (b[pi, p[j]] + b[p[i], pi] - b[pi, p[j]] - b[p[i], p[j]]))
                # note: b[p[i],pi]=b[pi,pi]; use explicit below for clarity
                # recompute correctly:
                d = int(ai[j] * (b[pi, p[j]] + b[p[i], pi] - b[pi, p[j]] - b[p[i], p[j]]))
                if d < best_delta:
                    best_delta = d
                    bi = i
                    bj = j
        if best_delta < 0:
            i, j = bi, bj
            pi, pj = p[i], p[j]
            # update C
            C[i] -= int(a[i, j] * (b[pi, pi] + b[pj, pj] - b[pi, pj] - b[pj, pi]))
            C[j] -= int(a[j, i] * (b[pj, pj] + b[pi, pi] - b[pj, pi] - b[pi, pj]))
            p[i], p[j] = pj, pi
            cur += best_delta
            improved = True
            if cur < best_cost:
                best_cost = cur
                best_p = p.copy()
        if not improved:
            break
    if cur < best_cost:
        best_cost = cur
        best_p = p.copy()
    return best_cost, best_p


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = len(a)
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s kill

    # --- Strong constructive start: rank matching ---
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    C0 = _compute_C(a, b, p0)
    best_cost = int(C0.sum())
    best_p = p0.copy()

    # First refinement on the rank-matching start
    p = p0.copy()
    C = C0.copy()
    best_cost, best_p = _refine(a, b, p, C, deadline, best_cost, best_p)

    # --- Random restarts for diversification ---
    rng = np.random.default_rng(12345)
    while time.time() < deadline - 2.0:
        p = rng.permutation(n).astype(np.int64)
        C = _compute_C(a, b, p)
        cur = int(C.sum())
        if cur < best_cost:
            best_cost = cur
            best_p = p.copy()
        best_cost, best_p = _refine(a, b, p, C, deadline, best_cost, best_p)

    return best_p.tolist()
# EVOLVE-BLOCK-END
