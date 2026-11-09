import numpy as np
import time
import random

def ring_routing() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

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

    # Precompute CW and CCW edge index sets for each demand.
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw = []
        edge = s
        while edge != t:
            cw.append(edge)
            edge = (edge + 1) % n_nodes
        cw_edges.append(cw)
        cw_set = set(cw)
        ccw_edges.append([e for e in range(n_nodes) if e not in cw_set])

    # Precompute integer weight vectors for exact load computation.
    cw_vecs = []
    ccw_vecs = []
    for i in range(D):
        w = demands[i][2]
        cw_v = [0] * n_nodes
        ccw_v = [0] * n_nodes
        for e in cw_edges[i]:
            cw_v[e] = w
        for e in ccw_edges[i]:
            ccw_v[e] = w
        cw_vecs.append(cw_v)
        ccw_vecs.append(ccw_v)

    # Hard lower bound on the optimal max edge load:
    # each edge must carry at least the sum of weights of demands that
    # necessarily traverse it (i.e., demands whose CW path uses the edge
    # and whose CCW path also uses the edge). For a ring, a demand uses
    # an edge in CW iff the edge is on the CW arc; it uses the edge in CCW
    # iff the edge is NOT on the CW arc. So a demand is forced onto an edge
    # only if it is on the CW arc AND on the CCW arc, which is impossible
    # (they are complementary). Hence the trivial lower bound is 0; a tighter
    # bound is the max over edges of the sum of weights of demands that use
    # that edge in at least one of the two routings is not a lower bound
    # (we can choose routing). Instead use: for each edge e, the load is at
    # least the max over demands of the weight if the demand is forced.
    # Since no demand is forced, use the simple bound: 0. We'll rely on
    # exhaustive search for the exact optimum.
    hard_lb = 0

    best_max_load = float('inf')
    best_mask = 0

    # Exact exhaustive search using integer arithmetic (fast, no numpy in loop).
    for mask in range(1 << D):
        if time.time() > deadline:
            break
        loads = [0] * n_nodes
        for i in range(D):
            if (mask >> i) & 1:
                v = ccw_vecs[i]
            else:
                v = cw_vecs[i]
            for e in range(n_nodes):
                loads[e] += v[e]
        ml = max(loads)
        if ml < best_max_load:
            best_max_load = ml
            best_mask = mask

    # If the exhaustive search found the hard lower bound, we are done.
    if best_max_load <= hard_lb:
        best_routing = np.array([1.0 if ((best_mask >> i) & 1) else 0.0 for i in range(D)])
        return best_routing

    # Local-search polish: only if time remains and we have not hit the hard bound.
    # This is a conditional improvement pass; it can only keep or improve the
    # best_max_load found by exhaustive search.
    if time.time() < deadline - 1.0:
        rng = random.Random(0)
        cur_mask = best_mask
        cur_load = best_max_load
        # Precompute per-edge load for current mask to enable O(n_nodes) flips.
        cur_loads = [0] * n_nodes
        for i in range(D):
            v = ccw_vecs[i] if ((cur_mask >> i) & 1) else cw_vecs[i]
            for e in range(n_nodes):
                cur_loads[e] += v[e]
        iters = 0
        max_iters = 200000
        while iters < max_iters and time.time() < deadline - 1.0:
            iters += 1
            # Try flipping each demand; accept the flip that most reduces max load.
            best_delta = 0
            best_flip = -1
            for i in range(D):
                # Compute new max load if we flip demand i.
                # Delta vector: new_vec - old_vec
                old_vec = ccw_vecs[i] if ((cur_mask >> i) & 1) else cw_vecs[i]
                new_vec = cw_vecs[i] if ((cur_mask >> i) & 1) else ccw_vecs[i]
                # Apply delta to a copy of loads, compute max.
                nl = cur_loads[:]
                for e in range(n_nodes):
                    nl[e] += new_vec[e] - old_vec[e]
                m = max(nl)
                d = m - cur_load
                if d < best_delta:
                    best_delta = d
                    best_flip = i
            if best_flip < 0:
                break
            # Apply the best flip.
            old_vec = ccw_vecs[best_flip] if ((cur_mask >> best_flip) & 1) else cw_vecs[best_flip]
            new_vec = cw_vecs[best_flip] if ((cur_mask >> best_flip) & 1) else ccw_vecs[best_flip]
            for e in range(n_nodes):
                cur_loads[e] += new_vec[e] - old_vec[e]
            cur_mask ^= (1 << best_flip)
            cur_load += best_delta
            if cur_load < best_max_load:
                best_max_load = cur_load
                best_mask = cur_mask
            if best_max_load <= hard_lb:
                break

    best_routing = np.array([1.0 if ((best_mask >> i) & 1) else 0.0 for i in range(D)])
    return best_routing
