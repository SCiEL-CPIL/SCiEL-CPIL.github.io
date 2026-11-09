# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solves the ring loading problem by exhaustively searching all 2^15 routing combinations
    to minimize the maximum edge load.
    """
    start_time = time.time()
    deadline = 630.0
    
    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute edges for CW and CCW routes for each demand
    # CW: s -> s+1 -> ... -> t (mod 10)
    # CCW: s -> s-1 -> ... -> t (mod 10)
    
    cw_edges = []
    ccw_edges = []
    weights = []
    
    for s, t, w in demands:
        weights.append(w)
        
        # Clockwise edges
        edges_cw = []
        current = s
        while current != t:
            edges_cw.append(current)
            current = (current + 1) % n_nodes
        cw_edges.append(edges_cw)
        
        # Counter-Clockwise edges
        edges_ccw = []
        current = s
        while current != t:
            edges_ccw.append((current - 1) % n_nodes)
            current = (current - 1) % n_nodes
        ccw_edges.append(edges_ccw)
        
    # Convert to numpy arrays for faster indexing if needed, but lists are fine for small N
    # We will use bitmasks to represent routing choices
    # Bit i = 0 means demand i is CW
    # Bit i = 1 means demand i is CCW
    
    best_max_load = float('inf')
    best_mask = 0
    
    # Iterate over all 2^15 combinations
    for mask in range(1 << n_demands):
        # Check time limit occasionally (though this loop is very fast)
        if (mask & 0xFF) == 0 and time.time() > deadline - 1.0:
            break
            
        loads = np.zeros(n_nodes)
        
        for i in range(n_demands):
            w = weights[i]
            if (mask >> i) & 1:
                # CCW
                for e in ccw_edges[i]:
                    loads[e] += w
            else:
                # CW
                for e in cw_edges[i]:
                    loads[e] += w
            
        current_max_load = np.max(loads)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_mask = mask
            
    # Construct the result array
    # 0.0 for CW, 1.0 for CCW (>= 0.5 routes CCW)
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0
            
    return result

# EVOLVE-BLOCK-END
