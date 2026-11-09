# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Minimize the maximum edge load on a 10-node ring with 15 demands.
    Exhaustively search all 2^15 routing assignments using incremental
    load updates, then return the best routing as a float array.
    """
    deadline = time.time() + 600.0  # safety margin
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]

    # Precompute CW edge masks (as Python ints) for each demand
    # CW from s to t uses edges s, s+1, ..., t-1 (mod N)
    cw_masks = []
    ccw_masks = []
    for (s, t, w) in demands:
        cw = 0
        e = s
        while e != t:
            cw |= (1 << e)
            e = (e + 1) % N
        ccw = 0
        e = (s - 1) % N
        while e != s:
            ccw |= (1 << e)
            e = (e - 1) % N
        cw_masks.append(cw)
        ccw_masks.append(ccw)

    # Initial state: all clockwise
    loads = np.zeros(N, dtype=np.float64)
    for i in range(D):
        s, t, w = demands[i]
        e = s
        while e != t:
            loads[e] += w
            e = (e + 1) % N

    best_max = float(np.max(loads))
    best_routing = np.zeros(D, dtype=np.float64)  # all 0 = all CW

    # Exhaustive DFS: flip each demand, recurse
    # Use iterative approach for speed
    # State: routing[i] = 0 means CW, 1 means CCW
    routing = np.zeros(D, dtype=np.int8)

    # Incremental flip: flipping demand i from CW to CCW:
    #   remove w from CW edges, add w to CCW edges
    # Flipping from CCW to CW: reverse
    # We track current direction per demand

    # For speed, use a recursive DFS with backtracking
    # But 2^15 = 32768 leaves, 32767 internal nodes — very fast

    def dfs(idx):
        nonlocal best_max, best_routing
        if idx == D:
            m = float(np.max(loads))
            if m < best_max:
                best_max = m
                best_routing = routing.astype(np.float64).copy()
            return

        # Option 1: keep current direction (no change)
        dfs(idx + 1)

        # Option 2: flip direction
        s, t, w = demands[idx]
        if routing[idx] == 0:
            # CW -> CCW: subtract w from CW edges, add w to CCW edges
            e = s
            while e != t:
                loads[e] -= w
                e = (e + 1) % N
            e = (s - 1) % N
            while e != s:
                loads[e] += w
                e = (e - 1) % N
        else:
            # CCW -> CW: add w to CW edges, subtract w from CCW edges
            e = s
            while e != t:
                loads[e] += w
                e = (e + 1) % N
            e = (s - 1) % N
            while e != s:
                loads[e] -= w
                e = (e - 1) % N
        routing[idx] ^= 1
        dfs(idx + 1)
        # Undo flip
        if routing[idx] == 0:
            # was CCW, now back to CW: add w to CW, subtract w from CCW
            e = s
            while e != t:
                loads[e] += w
                e = (e + 1) % N
            e = (s - 1) % N
            while e != s:
                loads[e] -= w
                e = (e - 1) % N
        else:
            # was CW, now back to CCW: subtract w from CW, add w to CCW
            e = s
            while e != t:
                loads[e] -= w
                e = (e + 1) % N
            e = (s - 1) % N
            while e != s:
                loads[e] += w
                e = (e - 1) % N
        routing[idx] ^= 1

    # Check time before starting exhaustive search
    if time.time() < deadline - 10.0:
        dfs(0)

    return best_routing

# EVOLVE-BLOCK-END
