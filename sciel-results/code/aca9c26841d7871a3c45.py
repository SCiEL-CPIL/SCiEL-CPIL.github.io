import numpy as np
import time

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
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_edges = 10
    D = len(demands)
    
    # Precompute base loads (all CW) and difference contributions
    base_load = [0.0] * n_edges
    diff_contrib = [[0.0] * n_edges for _ in range(D)]
    
    for i, (s, t, w) in enumerate(demands):
        # Determine CW edges
        cw_edges = []
        if s < t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        # Determine CCW edges (all other edges)
        cw_set = set(cw_edges)
        ccw_edges = [e for e in range(n_edges) if e not in cw_set]
        
        # Update base_load with CW contribution
        for e in cw_edges:
            base_load[e] += w
        
        # Update diff_contrib with (CCW - CW) contribution
        # If routed CCW, load adds w to ccw_edges.
        # If routed CW, load adds w to cw_edges.
        # diff = Load_CCW - Load_CW
        # So for e in ccw_edges: diff += w
        # For e in cw_edges: diff -= w
        for e in ccw_edges:
            diff_contrib[i][e] += w
        for e in cw_edges:
            diff_contrib[i][e] -= w

    best_max_load = float('inf')
    best_mask = 0
    total_masks = 1 << D
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 590.0 # 590s safety margin for 600s limit
    
    for mask in range(total_masks):
        if time.time() > deadline:
            break
            
        current_load = base_load[:]
        # Apply differences for demands routed CCW (bit set)
        for d in range(D):
            if (mask >> d) & 1:
                dc = diff_contrib[d]
                # Inline loop for speed
                current_load[0] += dc[0]
                current_load[1] += dc[1]
                current_load[2] += dc[2]
                current_load[3] += dc[3]
                current_load[4] += dc[4]
                current_load[5] += dc[5]
                current_load[6] += dc[6]
                current_load[7] += dc[7]
                current_load[8] += dc[8]
                current_load[9] += dc[9]
        
        max_load = max(current_load)
        if max_load < best_max_load:
            best_max_load = max_load
            best_mask = mask
    
    # Construct result array
    routing = np.zeros(D)
    for d in range(D):
        if (best_mask >> d) & 1:
            routing[d] = 1.0
        else:
            routing[d] = 0.0
            
    return routing

# EVOLVE-BLOCK-END
