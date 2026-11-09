import numpy as np

def ring_routing() -> np.ndarray:
    # Define demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute incidence matrices for CW and CCW
    # cw_mat[i, e] = weight of demand i if edge e is on CW path, else 0
    # ccw_mat[i, e] = weight of demand i if edge e is on CCW path, else 0
    cw_mat = np.zeros((n_demands, n_nodes), dtype=np.float64)
    ccw_mat = np.zeros((n_demands, n_nodes), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        # CW path
        curr = s
        while curr != t:
            cw_mat[i, curr] = w
            curr = (curr + 1) % n_nodes
            
        # CCW path
        curr = s
        while curr != t:
            curr = (curr - 1) % n_nodes
            ccw_mat[i, curr] = w

    # Generate all 2^15 masks as a binary matrix
    # M has shape (2^15, 15)
    # M[k, i] is 1 if demand i is routed CCW in mask k
    # We can generate this efficiently
    max_masks = 1 << n_demands
    # Use bitwise operations to generate the matrix
    # np.arange(max_masks) gives the mask integers
    masks = np.arange(max_masks)
    # Reshape to binary digits
    # We need the bits of the mask.
    # bit i corresponds to demand i.
    # masks is 1D.
    # We can use (masks[:, None] >> np.arange(n_demands)[None, :]) & 1
    M = (masks[:, None] >> np.arange(n_demands)[None, :]) & 1
    
    # Calculate loads for all masks simultaneously
    # loads[k, e] = sum over i of (M[k, i] * ccw_mat[i, e] + (1 - M[k, i]) * cw_mat[i, e])
    # loads = M @ ccw_mat + (1 - M) @ cw_mat
    # Note: M is int, mats are float. Result is float.
    
    # Optimization: loads = M @ (ccw_mat - cw_mat) + cw_mat
    # Let diff = ccw_mat - cw_mat
    # loads = M @ diff + cw_mat
    
    diff_mat = ccw_mat - cw_mat
    # M is (32768, 15), diff_mat is (15, 10)
    # M @ diff_mat is (32768, 10)
    loads_ccw_part = M @ diff_mat
    # Add base CW load (which is constant for all masks? No, cw_mat is constant, but (1-M) changes)
    # Wait, the formula `M @ (ccw - cw) + cw` is correct.
    # Because `M @ ccw + (1-M) @ cw = M @ ccw + cw - M @ cw = M @ (ccw - cw) + cw`
    
    total_loads = loads_ccw_part + cw_mat
    
    # Find max load per mask
    max_loads = np.max(total_loads, axis=1)
    
    # Find the mask with the minimum max load
    best_idx = np.argmin(max_loads)
    best_mask = masks[best_idx]
    
    # Convert best_mask back to routing array
    # If bit i is 1, route CCW (value >= 0.5, e.g., 1.0)
    # If bit i is 0, route CW (value < 0.5, e.g., 0.0)
    routing = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
