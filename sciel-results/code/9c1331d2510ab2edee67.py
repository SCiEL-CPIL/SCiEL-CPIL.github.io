# EVOLVE-BLOCK-START
import numpy as np


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        < 0.5 -> route the demand clockwise
        >= 0.5 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # The search space is 2^15 = 32768, which is negligible in time.
    # No deadline guard is needed, but we keep the structure simple.

    # Precompute masks.
    # Edge e is between node e and (e+1)%n.
    # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n).
    # Counter-clockwise: all other edges.
    cw_mask = np.zeros((D, n), dtype=np.float64)
    ccw_mask = np.zeros((D, n), dtype=np.float64)
    weights = np.zeros(D)
    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        if s == t:
            continue
        e = s
        while e != t:
            cw_mask[d, e] = 1.0
            e = (e + 1) % n
        ccw_mask[d, :] = 1.0 - cw_mask[d, :]

    # Weighted contribution of each demand routed CW or CCW.
    cw_w = cw_mask * weights[:, None]
    ccw_w = ccw_mask * weights[:, None]

    # base_load: load if every demand routed CW.
    base_load = cw_w.sum(axis=0)  # shape (n,)

    # diff[d, e] = ccw_w[d, e] - cw_w[d, e].
    # load(r) = base_load + r @ diff, where r[d] in {0,1} is the direction.
    diff = ccw_w - cw_w  # shape (D, n)

    # ---- Fast exhaustive search via Gray-code single-bit updates ----
    # Enumerate all 2^D combinations. Consecutive Gray codes differ in exactly
    # one bit, so we only add/subtract one diff row per step (O(n) per step).
    total = 1 << D  # 32768

    best_max_load = np.inf
    best_routing = np.zeros(D, dtype=np.float64)

    # Start at routing = 0 (all CW).
    load = base_load.copy()  # shape (n,)
    r = np.zeros(D, dtype=np.float64)
    cur_max = load.max()
    if cur_max < best_max_load:
        best_max_load = cur_max
        best_routing = r.copy()

    # Gray code: g = i ^ (i >> 1). The bit that changes between g(i-1) and g(i)
    # is the position of the least significant set bit of i.
    for i in range(1, total):
        bit = (i & (-i)).bit_length() - 1  # index of flipped bit
        if r[bit] == 0.0:
            r[bit] = 1.0
            load += diff[bit]
        else:
            r[bit] = 0.0
            load -= diff[bit]

        cur_max = load.max()
        if cur_max < best_max_load:
            best_max_load = cur_max
            best_routing = r.copy()

    # best_routing values are already 0.0 (CW) / 1.0 (CCW).
    # Contract: entry >= 0.5 -> CCW, else CW. Our 0.0/1.0 satisfy this.
    return best_routing.astype(float)


# EVOLVE-BLOCK-END
