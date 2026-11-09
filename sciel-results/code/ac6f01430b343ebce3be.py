# EVOLVE-BLOCK-START
import time
import numpy as np


def _build_data():
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)
    cw = []
    ccw = []
    for (s, t, w) in demands:
        if s == t:
            cw_edges = []
        else:
            cw_edges = []
            e = s
            while e != t:
                cw_edges.append(e)
                e = (e + 1) % n
        ccw_edges = [e for e in range(n) if e not in set(cw_edges)]
        cw.append((cw_edges, w))
        ccw.append((ccw_edges, w))
    return demands, cw, ccw, n, D


def ring_routing() -> np.ndarray:
    start = time.time()
    deadline = start + 590.0
    safety = 10.0

    demands, cw, ccw, n, D = _build_data()

    weights = [demands[i][2] for i in range(D)]

    # Precompute numpy arrays for load contributions
    contrib_cw = []
    contrib_ccw = []
    for i in range(D):
        vec_cw = np.zeros(n, dtype=float)
        vec_ccw = np.zeros(n, dtype=float)
        if len(cw[i][0]) > 0:
            vec_cw[cw[i][0]] = weights[i]
        if len(ccw[i][0]) > 0:
            vec_ccw[ccw[i][0]] = weights[i]
        contrib_cw.append(vec_cw)
        contrib_ccw.append(vec_ccw)

    # Stack into matrices for vectorized evaluation
    # contrib_cw_mat: shape (D, n) where row i is the contribution of demand i if routed CW
    # contrib_ccw_mat: shape (D, n) where row i is the contribution of demand i if routed CCW
    contrib_cw_mat = np.vstack(contrib_cw)
    contrib_ccw_mat = np.vstack(contrib_ccw)

    # Total load if all CW
    total_cw_load = contrib_cw_mat.sum(axis=0)
    # Delta load if switching demand i from CW to CCW: contrib_ccw[i] - contrib_cw[i]
    delta_mat = contrib_ccw_mat - contrib_cw_mat

    # ---- Phase 1: Exact global enumeration via Gray code ----
    best_load = float('inf')
    best_mask = 0
    total = 1 << D

    loads = total_cw_load.copy()
    prev_gray = 0
    best_load = float(np.max(loads))
    best_mask = 0

    for step in range(1, total):
        gray = step ^ (step >> 1)
        diff = gray ^ prev_gray
        bit = (diff & -diff).bit_length() - 1
        is_ccw = (gray >> bit) & 1

        if is_ccw:
            # was clockwise, now ccw
            loads += delta_mat[bit]
        else:
            # was ccw, now clockwise
            loads -= delta_mat[bit]
        prev_gray = gray

        cur = float(np.max(loads))
        if cur < best_load:
            best_load = cur
            best_mask = gray

        if time.time() > deadline - safety:
            break

    # ---- Phase 2: Local search polish (redundant if Phase 1 completed, but safe) ----
    routing = np.zeros(D)
    for i in range(D):
        routing[i] = 1.0 if (best_mask >> i) & 1 else 0.0

    def maxload_of(r):
        loads_arr = np.zeros(n)
        for i in range(D):
            if r[i] >= 0.5:
                loads_arr += contrib_ccw[i]
            else:
                loads_arr += contrib_cw[i]
        return float(np.max(loads_arr))

    cur_load = maxload_of(routing)
    improved = True
    while improved and (time.time() < deadline - safety):
        improved = False
        # single flips
        for i in range(D):
            trial = routing.copy()
            trial[i] = 0.0 if trial[i] >= 0.5 else 1.0
            v = maxload_of(trial)
            if v < cur_load - 1e-12:
                routing = trial
                cur_load = v
                improved = True
        # double flips
        for i in range(D):
            for j in range(i + 1, D):
                trial = routing.copy()
                trial[i] = 0.0 if trial[i] >= 0.5 else 1.0
                trial[j] = 0.0 if trial[j] >= 0.5 else 1.0
                v = maxload_of(trial)
                if v < cur_load - 1e-12:
                    routing = trial
                    cur_load = v
                    improved = True

    # Final exact verification
    final = maxload_of(routing)
    enum_routing = np.zeros(D)
    for i in range(D):
        enum_routing[i] = 1.0 if (best_mask >> i) & 1 else 0.0
    if maxload_of(enum_routing) < final - 1e-12:
        routing = enum_routing

    return routing


# EVOLVE-BLOCK-END
