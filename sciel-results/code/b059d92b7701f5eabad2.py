import numpy as np
import time

def ring_routing() -> np.ndarray:
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    cw_vec = np.zeros((n_demands, n_nodes), dtype=np.int8)
    ccw_vec = np.zeros((n_demands, n_nodes), dtype=np.int8)
    
    for i, (s, t, w) in enumerate(demands):
        current = s
        while current != t:
            cw_vec[i, current] = 1
            current = (current + 1) % n_nodes
        ccw_vec[i] = 1 - cw_vec[i]
        
    weights = np.array([w for s, t, w in demands], dtype=np.int32)
    diff = ccw_vec - cw_vec
    
    # Vectorized exhaustive search
    # 2^15 masks is small enough to process all at once
    n_masks = 1 << n_demands
    masks = np.arange(n_masks, dtype=np.uint16)
    
    # Generate direction matrix (n_masks, n_demands)
    # Bit i of mask determines direction for demand i
    # We want 1 if bit is set (CCW), 0 otherwise (CW)
    # Using bit shifts and masks
    # Note: np.arange(n_demands) creates [0, 1, ..., 14]
    # (masks[:, None] >> np.arange(n_demands)) & 1
    directions = ((masks[:, None] >> np.arange(n_demands)) & 1).astype(np.int8)
    
    # Calculate used edges for all masks
    # used shape: (n_masks, n_demands, n_nodes)
    # cw_vec shape: (n_demands, n_nodes) -> broadcast to (1, n_demands, n_nodes)
    # diff shape: (n_demands, n_nodes) -> broadcast to (1, n_demands, n_nodes)
    # directions shape: (n_masks, n_demands) -> (n_masks, n_demands, 1)
    
    used = cw_vec[None, :, :] + directions[:, :, None] * diff[None, :, :]
    
    # Calculate loads for all masks
    # loads shape: (n_masks, n_nodes)
    # used * weights[None, :, None] -> (n_masks, n_demands, n_nodes)
    # sum over axis 1 (demands)
    loads = np.sum(used * weights[None, :, None], axis=1)
    
    # Find mask with minimum max load
    max_loads = np.max(loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_routing = directions[best_idx].astype(np.float64)
    
    return best_routing
