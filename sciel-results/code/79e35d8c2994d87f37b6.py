# EVOLVE-BLOCK-START
import numpy as np


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute for each demand:
    #   cw_edges[d]: set of edges used if routed clockwise
    #   ccw_edges[d]: set of edges used if routed counter-clockwise
    # Edge e is between node e and (e+1)%n.
    # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n) if s != t
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
        # ccw is complement
        ccw_mask[d, :] = 1.0 - cw_mask[d, :]

    # Precompute weighted masks
    # cw_w[d, e] = weights[d] * cw_mask[d, e]
    # ccw_w[d, e] = weights[d] * ccw_mask[d, e]
    cw_w = cw_mask * weights[:, None]
    ccw_w = ccw_mask * weights[:, None]

    # base_load = sum over all demands routed CW
    base_load = cw_w.sum(axis=0)  # shape (n,)

    # diff[d, e] = ccw_w[d, e] - cw_w[d, e]
    # If demand d is routed CCW (r[d]=1), we add diff[d] to the load
    # load(r) = base_load + sum_d r[d] * diff[d]
    diff = ccw_w - cw_w  # shape (D, n)

    # Structural Change: Gray Code Enumeration with Incremental Updates
    # This avoids recomputing the full load vector from scratch for each bitmask.
    # Instead, we move from one bitmask to the next by flipping a single bit,
    # updating the load vector by adding or subtracting the corresponding diff vector.
    
    best_routing = np.zeros(D, dtype=np.int8)
    best_max_load = np.inf
    current_load = base_load.copy()
    prev_gray = 0
    
    # Precompute index of least significant set bit for quick bit identification
    # Or just use bit_length since we only have 1 bit changing
    for i in range(1 << D):
        gray = i ^ (i >> 1)
        if i > 0:
            changed_mask = prev_gray ^ gray
            # Find the index of the changed bit
            # Since only 1 bit changes, we can find it efficiently
            d = (changed_mask.bit_length() - 1)
            if gray & changed_mask:
                # Bit turned on -> add diff
                current_load += diff[d]
            else:
                # Bit turned off -> subtract diff
                current_load -= diff[d]
        
        max_load = current_load.max()
        if max_load < best_max_load:
            best_max_load = max_load
            # Convert gray code back to binary to store routing
            # Or we can just store the bits from gray? No, we need the actual routing bits.
            # The variable 'gray' represents the routing bits in Gray Code order.
            # We need the binary representation of 'gray' to match the demand order?
            # Wait, the 'gray' value IS the bitmask for the routing.
            # bit d of 'gray' is 1 if demand d is routed CCW.
            # So we can just extract bits from 'gray'.
            best_routing = np.array([(gray >> d) & 1 for d in range(D)], dtype=np.int8)
            
        prev_gray = gray

    return best_routing.astype(float)


# EVOLVE-BLOCK-END
