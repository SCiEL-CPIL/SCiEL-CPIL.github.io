import numpy as np
import time
from scipy.optimize import linprog

def ring_routing() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin (630s limit)

    n_nodes = 10
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]

    D = len(demands)

    # Precompute CW and CCW edge index sets for each demand
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw = []
        edge = s
        while edge != t:
            cw.append(edge)
            edge = (edge + 1) % n_nodes
        cw_edges.append(cw)
        ccw = [e for e in range(n_nodes) if e not in cw]
        ccw_edges.append(ccw)

    # Precompute the weight addition vectors for CW and CCW for each demand
    cw_vecs = []
    ccw_vecs = []
    for i in range(D):
        w = demands[i][2]
        cw_v = np.zeros(n_nodes)
        ccw_v = np.zeros(n_nodes)
        for e in cw_edges[i]:
            cw_v[e] = w
        for e in ccw_edges[i]:
            ccw_v[e] = w
        cw_vecs.append(cw_v)
        ccw_vecs.append(ccw_v)

    def compute_max_load(routing):
        """Compute max edge load given a binary routing array."""
        loads = np.zeros(n_nodes)
        for i in range(D):
            if routing[i] >= 0.5:
                loads += ccw_vecs[i]
            else:
                loads += cw_vecs[i]
        return np.max(loads)

    best_max_load = float('inf')
    best_routing = np.zeros(D)

    # Phase 1: Exhaustive search over all 2^15 = 32768 binary assignments
    for mask in range(1 << D):
        if time.time() > deadline:
            break

        loads = np.zeros(n_nodes)
        for i in range(D):
            if (mask >> i) & 1:
                loads += ccw_vecs[i]
            else:
                loads += cw_vecs[i]

        max_load = np.max(loads)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = np.array([1.0 if ((mask >> i) & 1) else 0.0 for i in range(D)])

    # Phase 2: Local search polish - try flipping bits to see if we can improve
    if time.time() < deadline:
        improved = True
        while improved:
            improved = False
            for i in range(D):
                trial = best_routing.copy()
                trial[i] = 1.0 - trial[i]
                trial_max = compute_max_load(trial)
                if trial_max < best_max_load:
                    best_max_load = trial_max
                    best_routing = trial
                    improved = True
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break

    # Phase 3: LP relaxation to get a lower bound and potential starting point
    if time.time() < deadline:
        base_loads = np.zeros(n_nodes)
        for i in range(D):
            base_loads += cw_vecs[i]

        A_ub = np.zeros((n_nodes, D + 1))
        b_ub = np.zeros(n_nodes)
        
        for e in range(n_nodes):
            b_ub[e] = -base_loads[e]
            for i in range(D):
                delta = ccw_vecs[i][e] - cw_vecs[i][e]
                A_ub[e, i] = delta
            A_ub[e, D] = -1.0

        bounds = [(0, 1)] * D + [(0, None)]
        c = np.zeros(D + 1)
        c[D] = 1.0

        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                lp_max_load = res.fun
                lp_x = res.x[:D]
                lp_routing = (lp_x >= 0.5).astype(float)
                lp_max = compute_max_load(lp_routing)
                if lp_max < best_max_load:
                    best_max_load = lp_max
                    best_routing = lp_routing
                    improved = True
                    while improved:
                        improved = False
                        for i in range(D):
                            trial = best_routing.copy()
                            trial[i] = 1.0 - trial[i]
                            trial_max = compute_max_load(trial)
                            if trial_max < best_max_load:
                                best_max_load = trial_max
                                best_routing = trial
                                improved = True
                            if time.time() > deadline:
                                break
                        if time.time() > deadline:
                            break
        except Exception:
            pass

    return best_routing
