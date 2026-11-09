# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem: 10-node ring, 15 demands,
    each routed CW or CCW to minimize max edge load.
    Uses exhaustive Gray-code enumeration with incremental load updates.
    2^15 = 32768 combinations is small enough for full enumeration.
    """
    start_time = time.time()
    DEADLINE = 600.0  # seconds

    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)

    # Precompute for each demand: the delta in edge loads when switching from CW to CCW
    # If demand d is routed CW, it uses edges s, s+1, ..., t-1 (mod n)
    # If routed CCW, it uses all other edges
    # delta[d][e] = (load change on edge e) when switching d from CW to CCW
    # = weight * (1 if e not in CW path) - weight * (1 if e in CW path)
    # = weight * (2*(1 if e not in CW path else 0) - 1)
    # So delta[d][e] = -w if e in CW path, +w if e not in CW path

    # base_loads: loads when all demands routed CW
    base_loads = [0] * n
    # deltas[d][e]: change in edge e load when flipping demand d from CW to CCW
    deltas = [[0] * n for _ in range(D)]

    for d, (s, t, w) in enumerate(demands):
        # Determine CW edges
        cw_edges = set()
        e = s
        while e != t:
            cw_edges.add(e % n)
            e = (e + 1) % n

        for edge in range(n):
            base_loads[edge] += w
            if edge in cw_edges:
                deltas[d][edge] = -w  # removing CW usage
            else:
                deltas[d][edge] = w   # adding CCW usage

    # Convert to lists for speed
    base_loads = list(base_loads)
    # deltas is already list of lists

    # Gray-code enumeration
    # Gray code: i XOR (i >> 1)
    # Consecutive Gray codes differ in exactly one bit
    # We track which bit changed to update loads incrementally

    best_val = float('inf')
    best_mask = 0

    # Start with all-CW (mask = 0)
    loads = list(base_loads)
    cur_max = max(loads)
    if cur_max < best_val:
        best_val = cur_max
        best_mask = 0

    # Enumerate all 2^D combinations via Gray code
    total = 1 << D  # 32768
    for i in range(1, total):
        if time.time() - start_time > DEADLINE - 5:
            break

        # Compute Gray code for i and i-1 to find which bit flipped
        g_prev = (i - 1) ^ ((i - 1) >> 1)
        g_curr = i ^ (i >> 1)
        diff_bits = g_prev ^ g_curr
        # Find the bit that changed
        bit = diff_bits & (-diff_bits)  # lowest set bit
        d = bit.bit_length() - 1  # 0-indexed demand

        # Update loads: flip demand d
        delta_d = deltas[d]
        for e in range(n):
            loads[e] += delta_d[e]

        cur_max = max(loads)
        if cur_max < best_val:
            best_val = cur_max
            best_mask = g_curr

    # Convert best_mask to routing vector
    # bit d of mask = 1 means demand d is routed CCW (value >= 0.5)
    # bit d of mask = 0 means demand d is routed CW (value < 0.5)
    routing = np.zeros(D, dtype=float)
    for d in range(D):
        if (best_mask >> d) & 1:
            routing[d] = 1.0
        else:
            routing[d] = 0.0

    return routing

# EVOLVE-BLOCK-END
