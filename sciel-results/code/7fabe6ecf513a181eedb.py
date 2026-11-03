# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _compute_cost(a, b, p):
    # exact integer cost
    return int(np.sum(a * b[np.ix_(p, p)]))


def solve(a, b):
    a = np.ascontiguousarray(a, dtype=np.int64)
    b = np.ascontiguousarray(b, dtype=np.int64)
    n = a.shape[0]
    t0 = time.time()
    deadline = t0 + 145.0  # safety margin before the 150s kill

    # ---- Initialization: linearization via Hungarian assignment ----
    # C[i, j] = -sum_k a[i,k] * b[j,k]  ->  b @ a.T
    C = b @ a.T
    # Hungarian minimizes sum C[i, p[i]] = -sum_k a[i,k]*b[p[i],k]
    # (captures the strongest facility-location affinities globally)
    try:
        ii, jj = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[ii] = jj
    except Exception:
        # fallback: rank matching
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy] = central

    # ---- Local search with incremental O(n) swap delta ----
    Bp = b[p]                       # Bp[k] = b[p[k]]
    colsum = Bp.sum(axis=0)         # colsum[j] = sum_k Bp[k, j]
    best_cost = _compute_cost(a, b, p)

    # precompute a's row structure for the delta
    # delta for swapping p[i], p[j]:
    #   D = sum_k a[i,k]*(Bp[k, pi] - Bp[k, pj]) + sum_k a[j,k]*(Bp[k, pj] - Bp[k, pi])
    #     + sum_k a[k,i]*(Bp[pj, k] - Bp[pi, k]) + sum_k a[k,j]*(Bp[pi, k] - Bp[pj, k])
    #   where pi=p[i], pj=p[j]

    order = np.arange(n)
    stall = 0
    while time.time() < deadline:
        improved = False
        # shuffle-free sweep; try all pairs in a rotating order for diversity
        for i in range(n - 1):
            if time.time() >= deadline:
                break
            ai = a[i]
            a_i_col = a[:, i]
            pi = p[i]
            for j in range(i + 1, n):
                pj = p[j]
                Bpi = Bp[pi]
                Bpj = Bp[pj]
                aj = a[j]
                a_j_col = a[:, j]
                # term1: flow from i
                t1 = ai @ (Bpi - Bpj)
                # term2: flow to j (from i's column? no) -> flow into j from i's row already; 
                # we need flow from j to i's old loc and i to j's old loc handled by col terms
                t2 = aj @ (Bpj - Bpi)
                # term3: flow from others to i (column i), loc changes pi->pj
                t3 = a_i_col @ (Bpj - Bpi)
                # term4: flow from others to j (column j), loc changes pj->pi
                t4 = a_j_col @ (Bpi - Bpj)
                delta = t1 + t2 + t3 + t4
                if delta < 0:
                    # apply swap
                    p[i], p[j] = pj, pi
                    Bp[pi], Bp[pj] = Bpj, Bpi
                    best_cost += int(delta)
                    improved = True
                    stall = 0
        if improved:
            continue
        # no improvement in full sweep -> restart with a perturbation if time remains
        stall += 1
        if stall >= 3:
            # random 2-opt perturbation to escape local optimum
            for _ in range(5):
                i = np.random.randint(0, n)
                j = np.random.randint(0, n)
                if i == j:
                    continue
                pi = p[i]; pj = p[j]
                p[i], p[j] = pj, pi
                Bp[pi], Bp[pj] = Bp[pj], Bp[pi]
            best_cost = _compute_cost(a, b, p)
            # rebuild colsum (unused in delta but keep consistent)
            stall = 0

    return p.tolist()
# EVOLVE-BLOCK-END
