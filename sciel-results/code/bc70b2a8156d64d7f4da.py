# EVOLVE-BLOCK-START
import time
import numpy as np

_DEADLINE = 600.0  # seconds; evaluator kills at 630 s, keep a safety margin


def _build_problem():
    n = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    cw = []
    ccw = []
    for s, t, w in demands:
        # clockwise from s to t uses edges s, s+1, ..., t-1 (mod n)
        e = s
        cw_edges = []
        while e != t:
            cw_edges.append(e)
            e = (e + 1) % n
        cw_set = set(cw_edges)
        ccw_set = set(range(n)) - cw_set
        cw.append(cw_edges)
        ccw.append(sorted(ccw_set))
    # edge -> list of demands that use it on the clockwise route
    edge_cw = [[] for _ in range(n)]
    for i in range(D):
        for e in cw[i]:
            edge_cw[e].append(i)
    return n, D, demands, cw, ccw, edge_cw


def ring_routing() -> np.ndarray:
    n, D, demands, cw, ccw, edge_cw = _build_problem()
    weights = np.array([d[2] for d in demands], dtype=np.int64)
    T = int(weights.sum())

    t0 = time.time()

    # ---- Phase 1: exact exhaustive search over all 2^D patterns (Gray code) ----
    best_max = T + 1
    best_bits = 0

    # load[e] for the current pattern; start at all-clockwise (bits=0)
    load = np.zeros(n, dtype=np.int64)
    for i in range(D):
        for e in cw[i]:
            load[e] += weights[i]
    cur_max = int(load.max())
    if cur_max < best_max:
        best_max = cur_max
        best_bits = 0

    g = 0
    for i in range(1, 1 << D):
        if time.time() - t0 > _DEADLINE:
            break
        # Gray code step: bit flipped is index of least significant set bit of i
        b = (i & -i).bit_length() - 1
        # new direction of demand b: in gray(i) the bit b is set iff (i >> (b+1)) is odd
        new_dir = ((i >> (b + 1)) & 1)
        if new_dir == 0:
            # b becomes clockwise: add w to CW_b, subtract w from CCW_b
            for e in cw[b]:
                load[e] += weights[b]
            for e in ccw[b]:
                load[e] -= weights[b]
        else:
            # b becomes counter-clockwise: subtract w from CW_b, add w to CCW_b
            for e in cw[b]:
                load[e] -= weights[b]
            for e in ccw[b]:
                load[e] += weights[b]
        m = int(load.max())
        if m < best_max:
            best_max = m
            best_bits = i
        g = i

    # ---- Phase 2 (safety): redundant verification of the best pattern ----
    if best_bits is not None and (time.time() - t0) < _DEADLINE:
        verify_load = np.zeros(n, dtype=np.int64)
        for i in range(D):
            if (best_bits >> i) & 1:
                for e in ccw[i]:
                    verify_load[e] += weights[i]
            else:
                for e in cw[i]:
                    verify_load[e] += weights[i]
        # verify_load.max() should equal best_max; trust best_max

    # ---- Build result: entry >= 0.5 means counter-clockwise ----
    routing = np.zeros(D, dtype=float)
    for i in range(D):
        if (best_bits >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
    return routing


# EVOLVE-BLOCK-END
