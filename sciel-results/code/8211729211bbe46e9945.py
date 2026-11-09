import numpy as np
import time

def ring_routing() -> np.ndarray:
    # Define the problem parameters
    n_nodes = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    n_demands = len(demands)
    
    # Precompute the edge usage vectors for Clockwise (CW) and Counter-Clockwise (CCW) paths
    # cw_edges[i, e] is 1 if demand i uses edge e when routed CW
    # ccw_edges[i, e] is 1 if demand i uses edge e when routed CCW
    cw_edges = np.zeros((n_demands, n_nodes), dtype=np.int8)
    ccw_edges = np.zeros((n_demands, n_nodes), dtype=np.int8)
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise path: s -> s+1 -> ... -> t
        cur = s
        while cur != t:
            cw_edges[i, cur] = 1
            cur = (cur + 1) % n_nodes
        
        # Counter-Clockwise path: s -> s-1 -> ... -> t
        # This is equivalent to all edges NOT in the CW path (excluding the destination edge if logic was different, 
        # but here CW covers s to t-1. CCW covers t to s-1. Together they cover all 10 edges? 
        # No, CW uses (t-s) edges. CCW uses (10-(t-s)) edges.
        # The union of CW and CCW edges for a pair (s,t) covers all 10 edges exactly once?
        # Yes, because the ring is fully connected. Edge e is either on the CW path or the CCW path.
        # So ccw_edges[i] = 1 - cw_edges[i] is correct for the set of edges used.
        ccw_edges[i] = 1 - cw_edges[i]
        
    weights = np.array([d[2] for d in demands], dtype=np.float64)
    
    # We need to choose a direction for each demand.
    # Let d_i = 0 for CW, 1 for CCW.
    # Total load on edge e = sum_i w_i * (cw_edges[i,e] * (1-d_i) + ccw_edges[i,e] * d_i)
    #                    = sum_i w_i * (cw_edges[i,e] + d_i * (ccw_edges[i,e] - cw_edges[i,e]))
    
    # Let A = cw_edges * weights (shape: n_demands x n_nodes)
    # Let B = (ccw_edges - cw_edges) * weights (shape: n_demands x n_nodes)
    # Load = A + D @ B, where D is a matrix of directions (n_masks x n_demands)
    
    A = cw_edges * weights[:, None]
    B = (ccw_edges - cw_edges) * weights[:, None]
    
    # Exhaustive search over all 2^15 combinations
    # This is feasible: 32,768 masks
    n_masks = 1 << n_demands
    
    # Generate all possible direction vectors
    # mask is an integer from 0 to 2^n_demands - 1
    # bit i of mask represents direction of demand i (0=CW, 1=CCW)
    masks = np.arange(n_masks, dtype=np.uint16)
    
    # Create direction matrix: shape (n_masks, n_demands)
    # directions[m, i] = (masks[m] >> i) & 1
    bits = np.arange(n_demands)
    directions = ((masks[:, None] >> bits[None, :]) & 1).astype(np.float64)
    
    # Calculate loads for all masks
    # loads shape: (n_masks, n_nodes)
    # loads = directions @ B + A[None, :, :]
    loads = directions @ B + A[None, :, :]
    
    # Objective: minimize the maximum edge load
    max_loads = np.max(loads, axis=1)
    
    # Find the mask with the minimum max load
    best_idx = np.argmin(max_loads)
    best_directions = directions[best_idx]
    
    # The return value should be an array of 15 numbers.
    # Entry >= 0.5 routes CCW, < 0.5 routes CW.
    # Our directions are 0 or 1. 1 is CCW, 0 is CW.
    # So we can return the directions directly.
    return best_directions

