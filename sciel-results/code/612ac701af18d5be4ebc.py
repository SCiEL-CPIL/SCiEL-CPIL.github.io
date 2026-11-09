import numpy as np

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
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
    
    # Precompute edge lists for CW and CCW for each demand
    cw_edges = []
    ccw_edges = []
    weights = []
    
    for s, t, w in demands:
        cw_e = []
        curr = s
        while curr != t:
            cw_e.append(curr)
            curr = (curr + 1) % N
        cw_edges.append(cw_e)
        
        ccw_e = []
        curr = s
        while curr != t:
            curr = (curr - 1) % N
            ccw_e.append(curr)
        ccw_edges.append(ccw_e)
        weights.append(w)

    # Create matrices for vectorized load calculation
    # shape (D, N)
    cw_matrix = np.zeros((D, N))
    ccw_matrix = np.zeros((D, N))
    
    for i in range(D):
        w = weights[i]
        for e in cw_edges[i]:
            cw_matrix[i, e] = w
        for e in ccw_edges[i]:
            ccw_matrix[i, e] = w

    best_max_load = float('inf')
    best_mask = 0
    total_masks = 1 << D
    
    # Vectorized brute force
    for mask in range(total_masks):
        # Create a boolean array of shape (D,) indicating if bit i is set
        # mask_bits[i] is True if demand i is CCW
        mask_bits = np.array([(mask >> i) & 1 for i in range(D)], dtype=bool)
        
        # Select matrices based on mask
        # where mask_bits is True, use ccw_matrix, else cw_matrix
        # We want to sum over demands (axis 0) to get edge loads (shape N)
        
        # Efficient way:
        # loads = np.sum(np.where(mask_bits[:, None], ccw_matrix, cw_matrix), axis=0)
        
        # Even faster:
        # loads = np.sum(cw_matrix, axis=0)
        # for i in range(D):
        #     if mask_bits[i]:
        #         loads += ccw_matrix[i] - cw_matrix[i]
        
        # Let's stick to the np.where approach, it's clean and fast enough for 32k iterations.
        selected = np.where(mask_bits[:, None], ccw_matrix, cw_matrix)
        loads = np.sum(selected, axis=0)
        
        current_max = np.max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = mask

    # Construct result
    routing = np.zeros(D)
    for i in range(D):
        if (best_mask >> i) & 1:
            routing[i] = 1.0 # CCW
        else:
            routing[i] = 0.0 # CW
            
    return routing
