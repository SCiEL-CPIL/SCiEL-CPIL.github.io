# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    """
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin before 630s kill

    N = 10  # nodes / edges
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute edge sets for each demand in each direction
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod N)
    # Counter-clockwise uses all other edges
    cw_edges = []
    ccw_edges = []
    for (s, t, w) in demands:
        cw = set()
        e = s
        while e != t:
            cw.add(e)
            e = (e + 1) % N
        all_edges = set(range(N))
        ccw = all_edges - cw
        cw_edges.append(cw)
        ccw_edges.append(ccw)

    weights = np.array([w for (_, _, w) in demands], dtype=float)

    def compute_loads(routing):
        """routing: array of 0 (cw) or 1 (ccw) per demand. Returns edge loads."""
        loads = np.zeros(N, dtype=float)
        for d in range(D):
            if routing[d] >= 0.5:  # ccw
                for e in ccw_edges[d]:
                    loads[e] += weights[d]
            else:  # cw
                for e in cw_edges[d]:
                    loads[e] += weights[d]
        return loads

    def max_load(routing):
        return compute_loads(routing).max()

    # Build edge-demand incidence matrices for speed
    # cw_matrix[d, e] = 1 if demand d uses edge e when routed cw
    cw_matrix = np.zeros((D, N))
    ccw_matrix = np.zeros((D, N))
    for d in range(D):
        for e in cw_edges[d]:
            cw_matrix[d, e] = 1.0
        for e in ccw_edges[d]:
            ccw_matrix[d, e] = 1.0

    def compute_loads_fast(routing_binary):
        """routing_binary: 0=cw, 1=ccw. Returns edge loads via matrix ops."""
        # loads = cw_matrix.T * (w * (1-r)) + ccw_matrix.T * (w * r)
        r = routing_binary.astype(float)
        cw_contrib = (weights * (1.0 - r)) @ cw_matrix  # shape (N,)
        ccw_contrib = (weights * r) @ ccw_matrix
        return cw_contrib + ccw_contrib

    # Exhaustive search: 2^15 = 32768 possibilities - feasible!
    best_routing = None
    best_max = float('inf')

    # Use numpy to vectorize over all 2^15 combinations
    # Actually, let's do it efficiently: iterate but with fast load computation
    # 32768 iterations with matrix ops should be fast enough

    total_combos = 1 << D
    for mask in range(total_combos):
        if time.time() > deadline:
            break
        routing = np.array([(mask >> d) & 1 for d in range(D)], dtype=float)
        loads = compute_loads_fast(routing)
        ml = loads.max()
        if ml < best_max:
            best_max = ml
            best_routing = routing.copy()

    # Return best_routing as 0/1 values (0 = cw, 1 = ccw)
    # The task says >= 0.5 routes ccw, so 1.0 works for ccw, 0.0 for cw
    return best_routing

# EVOLVE-BLOCK-END
