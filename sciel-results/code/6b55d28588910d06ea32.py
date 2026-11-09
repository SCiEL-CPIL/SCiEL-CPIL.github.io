import numpy as np

def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)

    # Precompute contributions
    # edge_contrib_cw[d, e] = weight if edge e is on CW path, else 0
    edge_contrib_cw = np.zeros((D, n), dtype=float)
    edge_contrib_ccw = np.zeros((D, n), dtype=float)

    for d, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            edge_contrib_cw[d, e % n] = w
            e = (e + 1) % n
        for e in range(n):
            if edge_contrib_cw[d, e] == 0:
                edge_contrib_ccw[d, e] = w

    base_cw_load = edge_contrib_cw.sum(axis=0)
    delta = edge_contrib_ccw - edge_contrib_cw

    best_val = float('inf')
    best_routing = np.zeros(D, dtype=float)

    # Exhaustive search
    for mask in range(1 << D):
        loads = base_cw_load.copy()
        for d in range(D):
            if (mask >> d) & 1:
                loads += delta[d]
        ml = float(np.max(loads))
        if ml < best_val:
            best_val = ml
            best_routing = np.zeros(D, dtype=float)
            for d in range(D):
                if (mask >> d) & 1:
                    best_routing[d] = 1.0

    return best_routing
