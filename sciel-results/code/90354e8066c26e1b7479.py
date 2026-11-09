# EVOLVE-BLOCK-START
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
    D = 15
    N = 10
    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute edge masks for each demand for CW and CCW routing
    # cw_edges[i] is a boolean array of length N indicating which edges are used if demand i goes CW
    # ccw_edges[i] is a boolean array of length N indicating which edges are used if demand i goes CCW
    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        cw_mask = np.zeros(N, dtype=bool)
        ccw_mask = np.zeros(N, dtype=bool)
        
        # Clockwise from s to t: edges s, s+1, ..., t-1 (mod N)
        # If s == t, no edges? But in ring routing problems s != t usually.
        # Let's assume standard definition: path from s to t in direction +1.
        # Edges are indexed by their start node. Edge e connects e to (e+1)%N.
        
        # Calculate number of steps clockwise
        steps_cw = (t - s) % N
        if steps_cw == 0:
             # If source == target, load is 0? Or does it traverse whole ring?
             # Standard interpretation: 0 distance.
             pass
        else:
            for i in range(steps_cw):
                e = (s + i) % N
                cw_mask[e] = True
        
        # Counter-clockwise from s to t: edges t-1, t-2, ..., s+1 (mod N)
        # Or equivalently, all edges NOT in the CW path (if s != t)
        # Let's just compute it directly to be safe.
        steps_ccw = (s - t) % N
        if steps_ccw == 0:
            pass
        else:
            for i in range(steps_ccw):
                e = (s - 1 - i) % N
                ccw_mask[e] = True
                
        cw_edges.append(cw_mask)
        ccw_edges.append(ccw_mask)
        
    # Convert to numpy array for faster operations if needed, but list of lists is fine
    # cw_edges is a list of 15 arrays of size 10
    
    best_load = float('inf')
    best_routing = np.zeros(D, dtype=int)
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 500.0 # 500s limit, well under 600s
    
    # Iterate over all 2^15 combinations
    for mask in range(1 << D):
        if time.time() > deadline:
            break
            
        loads = np.zeros(N, dtype=float)
        
        for i in range(D):
            if (mask >> i) & 1:
                # Route CCW
                # Add weight to edges in ccw_edges[i]
                loads[ccw_edges[i]] += demands[i][2]
            else:
                # Route CW
                # Add weight to edges in cw_edges[i]
                loads[cw_edges[i]] += demands[i][2]
                
        current_max_load = np.max(loads)
        
        if current_max_load < best_load:
            best_load = current_max_load
            # Decode mask to routing array
            # We need to return 0 for CW, 1 for CCW
            # Our mask bit i is 1 for CCW, 0 for CW. This matches the requirement.
            best_routing = np.array([(mask >> i) & 1 for i in range(D)], dtype=float)
            
    return best_routing

# EVOLVE-BLOCK-END
