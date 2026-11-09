# EVOLVE-BLOCK-START
import time
import numpy as np


def _build_data():
    n = 10
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
    d = len(demands)
    cw = np.zeros((d, n), dtype=np.int8)
    ccw = np.zeros((d, n), dtype=np.int8)
    w = np.zeros(d, dtype=np.int64)
    for i, (s, t, wt) in enumerate(demands):
        w[i] = wt
        if s == t:
            continue
        e = s
        while e != t:
            cw[i, e] = 1
            e = (e + 1) % n
        ccw[i] = 1 - cw[i]
    return n, d, cw, ccw, w


def ring_routing() -> np.ndarray:
    n, d, cw, ccw, w = _build_data()
    total = int(w.sum())
    # initial: all clockwise (0)
    routing = np.zeros(d, dtype=np.int8)
    loads = cw.T @ w  # shape (n,)
    cur_max = int(loads.max())

    deadline = time.time() + 580.0
    # best known
    best_routing = routing.copy()
    best_max = cur_max

    # vectorized flip evaluation
    # delta_cw_to_ccw = w[:, None] * (cw - ccw).T  # shape (n, d)
    delta = (w[None, :] * (cw - ccw).T).astype(np.int64)  # (n, d)

    # local search: flip demand that most reduces max load
    # If max is unique, flipping the demand on that edge with largest delta reduces max the most.
    # If multiple max edges, flipping a demand on any of them with largest delta may help.
    # We'll do: for each demand, compute new_max if flipped, pick best.
    # But that's O(d*n) per step which is fine for d=15, n=10.
    # More efficient: only consider demands on current max edges.
    max_edges = np.where(loads == cur_max)[0]

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # find best flip among demands on max edges
        best_gain = 0
        best_idx = -1
        for e in max_edges:
            # demands that use edge e when clockwise
            # flipping them removes w from e
            # we want to maximize reduction in max load
            # candidate demands: those with cw[i, e] == 1 (currently clockwise on e)
            cand = np.where(cw[:, e] == 1)[0]
            if len(cand) == 0:
                continue
            for i in cand:
                # simulate flip
                new_loads = loads + delta[:, i]
                nm = int(new_loads.max())
                gain = cur_max - nm
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
        if best_idx >= 0:
            # apply flip
            routing[best_idx] = 1 - routing[best_idx]
            loads = loads + delta[:, best_idx]
            cur_max = int(loads.max())
            if cur_max < best_max:
                best_max = cur_max
                best_routing = routing.copy()
            improved = True
            max_edges = np.where(loads == cur_max)[0]
        else:
            # no single flip improves max; try 2-flips?
            # For simplicity, stop single-flip local search
            # Try random restarts if time permits
            break

    # If we have time, do random restarts with local search
    if time.time() < deadline - 60:
        rng = np.random.default_rng(42)
        for restart in range(50):
            if time.time() > deadline - 30:
                break
            routing = rng.integers(0, 2, size=d).astype(np.int8)
            loads = np.zeros(n, dtype=np.int64)
            for i in range(d):
                if routing[i] == 0:
                    loads += w[i] * cw[i]
                else:
                    loads += w[i] * ccw[i]
            cur_max = int(loads.max())
            if cur_max < best_max:
                best_max = cur_max
                best_routing = routing.copy()
            # local search from here
            max_edges = np.where(loads == cur_max)[0]
            improved = True
            while improved and time.time() < deadline - 30:
                improved = False
                best_gain = 0
                best_idx = -1
                for e in max_edges:
                    cand = np.where(cw[:, e] == 1)[0]
                    if len(cand) == 0:
                        continue
                    for i in cand:
                        new_loads = loads + delta[:, i]
                        nm = int(new_loads.max())
                        gain = cur_max - nm
                        if gain > best_gain:
                            best_gain = gain
                            best_idx = i
                if best_idx >= 0:
                    routing[best_idx] = 1 - routing[best_idx]
                    loads = loads + delta[:, best_idx]
                    cur_max = int(loads.max())
                    if cur_max < best_max:
                        best_max = cur_max
                        best_routing = routing.copy()
                    improved = True
                    max_edges = np.where(loads == cur_max)[0]

    return best_routing.astype(np.float64)


# EVOLVE-BLOCK-END
