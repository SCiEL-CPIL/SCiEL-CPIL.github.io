# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem.
    Uses brute-force enumeration (2^15 = 32768 possibilities) to guarantee the
    globally optimal minimum maximum edge load. This is fast enough to run
    multiple times or with overhead well within the time limit.
    """
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
        (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute edge sets for each demand in each direction
    # cw_matrix[i, e] = weight if demand i is routed clockwise, 0 otherwise
    # ccw_matrix[i, e] = weight if demand i is routed counter-clockwise, 0 otherwise
    cw_matrix = np.zeros((D, n_nodes))
    ccw_matrix = np.zeros((D, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise edges: s, s+1, ..., t-1 (mod 10)
        if s <= t:
            cw_edges = range(s, t)
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        for e in cw_edges:
            cw_matrix[i, e] = w
            
        # Counter-clockwise edges: all edges NOT in clockwise path
        for e in range(n_nodes):
            if cw_matrix[i, e] == 0:
                ccw_matrix[i, e] = w

    # Brute force search
    # We want to find routing vector r in {0, 1}^D that minimizes max(loads)
    # loads = cw_matrix.sum(axis=0) + r @ (ccw_matrix - cw_matrix)
    
    base_loads = cw_matrix.sum(axis=0)
    delta_matrix = ccw_matrix - cw_matrix
    
    best_max_load = float('inf')
    best_routing = np.zeros(D)
    
    # Iterate through all 2^D combinations
    # Using bit manipulation for speed
    for mask in range(1 << D):
        # Construct routing vector from mask
        # Bit i of mask corresponds to demand i
        # If bit is 1, route CCW (1), else CW (0)
        r = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.float64)
        
        # Calculate loads
        loads = base_loads + r @ delta_matrix
        current_max = np.max(loads)
        
        if current_max < best_max_load:
            best_max_load = current_max
            best_routing = r
            
            # Early termination if we hit the theoretical lower bound?
            # The sum of all weights is fixed. The total load on all edges is sum(weights) * (path_length_avg?) No.
            # Total load = sum over edges of load(e).
            # For each demand, it contributes w * (path_length).
            # Min path length is min(dist_cw, dist_ccw).
            # This lower bound is complex to compute dynamically.
            # Given 32768 is small, we just run all.
            
    return best_routing

# EVOLVE-BLOCK-END
