# EVOLVE-BLOCK-START
import time
import numpy as np


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges.
    A FIXED set of D = 15 demands is each routed clockwise (0) or
    counter-clockwise (1) to MINIMIZE the maximum load over the 10 edges.

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
        entry >= 0.5 -> counter-clockwise, else clockwise.
    """
    t_start = time.time()
    DEADLINE = 620.0  # safety margin vs 630s wall-clock kill

    # ---- Problem data ----
    n = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]

    # Clockwise incidence: for demand d, edges on the clockwise path
    A = np.zeros((D, n), dtype=np.int64)
    w = np.zeros(D, dtype=np.float64)
    for d, (s, t, wt) in enumerate(demands):
        e = s
        while e != t:
            A[d, e] = 1
            e = (e + 1) % n
        w[d] = wt

    # Counter-clockwise load contribution for demand d on edge e:
    # it equals w[d] iff edge e is NOT on the clockwise path of d
    # (ccw path uses all other edges).
    # We represent routing as x[d] in {0,1}; 0=cw, 1=ccw.
    # Load on edge e = sum_d w[d] * ( A[d,e] if x[d]==0 else (1 - A[d,e]) )
    #                = sum_d w[d] * ( A[d,e] + x[d]*(1 - 2*A[d,e]) )
    # = base[e] + sum_d x[d] * w[d]*(1 - 2*A[d,e])
    base = (A * w[:, None]).sum(axis=0)  # shape (n,)
    C = w[:, None] * (1 - 2 * A)         # shape (D, n)  delta per flip to ccw

    def loads_from_x(x):
        # x: (D,) 0/1
        return base + x @ C

    def max_load(x):
        return float(np.max(loads_from_x(x)))

    # ---- Deterministic greedy initialization ----
    def greedy_init(seed):
        x = np.zeros(D, dtype=np.int64)
        # order demands by weight descending for a decent start
        order = np.argsort(-w)
        for d in order:
            # try both, pick lower max load
            x[d] = 0
            l0 = max_load(x)
            x[d] = 1
            l1 = max_load(x)
            x[d] = 0 if l0 <= l1 else 1
        return x

    # ---- Local search: single-flip best improvement ----
    def local_search(x, max_iters=200000):
        x = x.copy()
        cur = loads_from_x(x)
        cur_max = float(np.max(cur))
        for it in range(max_iters):
            best_delta = 0.0
            best_d = -1
            # For flipping d: load changes by C[d] if going 0->1, -C[d] if 1->0
            # Evaluate new max for each candidate flip.
            for d in range(D):
                if x[d] == 0:
                    new_loads = cur + C[d]
                else:
                    new_loads = cur - C[d]
                nm = float(np.max(new_loads))
                # improvement is negative of (nm - cur_max); we want to reduce cur_max
                if nm < cur_max - 1e-12:
                    # prefer larger reduction
                    if nm < best_delta or best_d == -1:
                        best_delta = nm
                        best_d = d
            if best_d == -1:
                break
            # apply flip
            x[best_d] = 1 - x[best_d]
            cur = loads_from_x(x)
            cur_max = best_delta
        return x, cur_max

    # ---- Multi-start local search ----
    best_x = greedy_init(0)
    best_val = max_load(best_x)

    starts = [best_x]
    rng = np.random.default_rng(12345)
    for r in range(200):
        x0 = np.zeros(D, dtype=np.int64)
        x0[rng.integers(0, 2, size=D)] = 1
        starts.append(x0)

    for x0 in starts:
        if time.time() > DEADLINE - 30:
            break
        x, val = local_search(x0)
        if val < best_val - 1e-12:
            best_val = val
            best_x = x

    # ---- Optional smooth-surrogate polish (log-sum-exp) ----
    # Try to improve using a smooth approximation of max load via a small
    # coordinate-descent on a continuous relaxation, then round.
    if time.time() < DEADLINE - 10:
        def smooth_obj(x, beta=20.0):
            L = loads_from_x(x.astype(np.float64))
            m = np.max(L)
            # log-sum-exp centered at m
            s = np.log(np.sum(np.exp(beta * (L - m)))) / beta + m
            return float(s)

        # coordinate descent on continuous x in [0,1]
        xc = best_x.astype(np.float64).copy()
        cur_s = smooth_obj(xc)
        for it in range(20000):
            if time.time() > DEADLINE - 10:
                break
            improved = False
            for d in range(D):
                # try both 0 and 1 (and current)
                vals = [cur_s]
                cand = [xc.copy()]
                if xc[d] != 0.0:
                    tmp = xc.copy(); tmp[d] = 0.0
                    vals.append(smooth_obj(tmp)); cand.append(tmp)
                if xc[d] != 1.0:
                    tmp = xc.copy(); tmp[d] = 1.0
                    vals.append(smooth_obj(tmp)); cand.append(tmp)
                bi = int(np.argmin(vals))
                if bi != 0 and vals[bi] < cur_s - 1e-12:
                    xc = cand[bi]
                    cur_s = vals[bi]
                    improved = True
            if not improved:
                break
        # round and local-search
        xr = np.where(xc >= 0.5, 1, 0).astype(np.int64)
        xr, v = local_search(xr)
        if v < best_val - 1e-12:
            best_val = v
            best_x = xr

    # ---- Return ----
    # entry >= 0.5 routes counter-clockwise. Use 1.0 for ccw, 0.0 for cw.
    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
