# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
        0 -> clockwise, >= 0.5 -> counter-clockwise
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin before 630s limit

    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute masks
    # cw_mask[d, e] = 1 if edge e is used when routing demand d clockwise
    # ccw_mask[d, e] = 1 if edge e is used when routing demand d counter-clockwise
    cw_mask = np.zeros((D, n), dtype=np.int32)
    ccw_mask = np.zeros((D, n), dtype=np.int32)
    weights = np.zeros(D, dtype=np.float64)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        if s == t:
            continue
        e = s
        while e != t:
            cw_mask[d, e] = 1
            e = (e + 1) % n
        
        # Counter-clockwise uses all other edges
        # Since it's a ring, ccw edges are the complement of cw edges
        ccw_mask[d, :] = 1 - cw_mask[d, :]

    # Precompute base load (all CW) and delta (change when flipping to CCW)
    # Load_e = sum_d w_d * ( (1-x_d)*cw_mask[d,e] + x_d*ccw_mask[d,e] )
    #        = sum_d w_d * cw_mask[d,e] + x_d * w_d * (ccw_mask[d,e] - cw_mask[d,e])
    
    cw_contrib = cw_mask * weights[:, np.newaxis]
    ccw_contrib = ccw_mask * weights[:, np.newaxis]
    
    base_load = cw_contrib.sum(axis=0)
    delta = ccw_contrib - cw_contrib  # Shape (D, n)

    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=np.float64)
    
    # Brute force all 2^D combinations
    # D=15, so 32768 iterations. Very fast.
    for mask in range(1 << D):
        if time.time() > deadline:
            break
            
        # Determine which demands are CCW (bit is 1)
        # Create boolean mask of shape (D,)
        # np.arange(D) is [0, 1, ..., D-1]
        # (mask >> np.arange(D)) & 1 gives the bit for each demand
        flips = ((mask >> np.arange(D)) & 1).astype(bool)
        
        # Compute loads
        # Sum deltas for flipped demands
        if np.any(flips):
            load_delta = delta[flips].sum(axis=0)
        else:
            load_delta = 0.0
            
        loads = base_load + load_delta
        max_load = np.max(loads)
        
        if max_load < best_max_load:
            best_max_load = max_load
            # Convert mask to routing array
            # 0 -> CW, 1 -> CCW
            best_routing = ((mask >> np.arange(D)) & 1).astype(np.float64)

    return best_routing

# EVOLVE-BLOCK-END
