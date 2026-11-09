import numpy as np
import itertools

def ring_routing() -> np.ndarray:
    # Define the ring parameters
    n_nodes = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    num_demands = len(demands)
    
    # Precompute edge masks for each demand
    # mask[d][e] is 1 if demand d uses edge e when routed clockwise, 0 otherwise
    # If routed counter-clockwise, it uses the complement edges (excluding the direct connection? No, "all other ring edges")
    # The ring has 10 edges.
    # Clockwise path from s to t: edges s, (s+1)%10, ..., (t-1)%10.
    # Counter-clockwise path: the rest of the edges.
    
    cw_masks = []
    ccw_masks = []
    for s, t, w in demands:
        cw_mask = np.zeros(n_nodes)
        # Traverse clockwise from s to t
        current = s
        while current != t:
            cw_mask[current] = 1
            current = (current + 1) % n_nodes
        
        # CCW uses all other edges
        ccw_mask = 1 - cw_mask
        cw_masks.append(cw_mask)
        ccw_masks.append(ccw_mask)
        
    # Convert to list of lists for faster indexing in loop if needed, or keep as np arrays
    # Let's use numpy broadcasting for vectorization if possible, but 32k iterations is fine with simple loops.
    
    best_max_load = float('inf')
    best_routing = np.zeros(num_demands)
    
    # Exhaustive search over all 2^15 combinations
    for combo in itertools.product([0, 1], repeat=num_demands):
        # combo[i] is 0 for CW, 1 for CCW
        loads = np.zeros(n_nodes)
        for i in range(num_demands):
            if combo[i] == 0:
                loads += demands[i][2] * cw_masks[i]
            else:
                loads += demands[i][2] * ccw_masks[i]
        
        current_max_load = np.max(loads)
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_routing = np.array(combo, dtype=float)
            
    return best_routing
