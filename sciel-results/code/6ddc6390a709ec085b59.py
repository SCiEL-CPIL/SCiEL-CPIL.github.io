import numpy as np
import time

def ring_routing() -> np.ndarray:
    t_start = time.time()
    deadline = t_start + 600.0  # 600s wall-clock limit; use 580s as safety margin
    safety_deadline = t_start + 580.0
    
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
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute CW and CCW load vectors for each demand
    cw_loads = []
    ccw_loads = []
    
    for s, t, w in demands:
        # CW path from s to t: edges s, s+1, ..., t-1 (mod 10)
        cw_edges = []
        if s == t:
            cw_edges = []
        elif s < t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        cw_set = set(cw_edges)
        ccw_edges = [e for e in range(n_nodes) if e not in cw_set]
        
        cw_vec = np.zeros(n_nodes)
        for e in cw_edges:
            cw_vec[e] = w
            
        ccw_vec = np.zeros(n_nodes)
        for e in ccw_edges:
            ccw_vec[e] = w
            
        cw_loads.append(cw_vec)
        ccw_loads.append(ccw_vec)
    
    cw_arr = np.array(cw_loads)  # shape (15, 10)
    ccw_arr = np.array(ccw_loads)  # shape (15, 10)
    
    # Base load if all demands are routed CW
    base = np.sum(cw_arr, axis=0)  # shape (10,)
    # Delta if demand i is switched from CW to CCW
    deltas = ccw_arr - cw_arr  # shape (15, 10)
    
    # Vectorized exhaustive search
    # Number of masks is 2^15 = 32768
    num_masks = 1 << n_demands
    
    # Create a matrix of bits: shape (num_masks, n_demands)
    # bits[m, i] is 1 if bit i of mask m is set
    masks = np.arange(num_masks)
    # Use broadcasting to extract bits
    # masks[:, None] has shape (num_masks, 1)
    # np.arange(n_demands)[None, :] has shape (1, n_demands)
    bits = ((masks[:, None] >> np.arange(n_demands)[None, :]) & 1).astype(np.float64)
    
    # Compute loads for all masks: shape (num_masks, n_nodes)
    # loads = base + bits @ deltas
    # base is (10,), bits is (32768, 15), deltas is (15, 10)
    # bits @ deltas is (32768, 10)
    loads = base[None, :] + (bits @ deltas)
    
    # Find the mask with the minimum maximum load
    max_loads = np.max(loads, axis=1)
    best_idx = np.argmin(max_loads)
    best_mask = int(masks[best_idx])
    best_max_load = float(max_loads[best_idx])
    
    # Local search polish: try single-flip and double-flip moves
    # Current routing vector (0=CW, 1=CCW)
    current_bits = np.zeros(n_demands, dtype=np.float64)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            current_bits[i] = 1.0
    
    current_loads = base + current_bits @ deltas
    current_max = float(np.max(current_loads))
    
    # Single-flip local search
    improved = True
    while improved and time.time() < safety_deadline:
        improved = False
        for i in range(n_demands):
            # Try flipping bit i
            new_bits = current_bits.copy()
            new_bits[i] = 1.0 - new_bits[i]
            new_loads = base + new_bits @ deltas
            new_max = float(np.max(new_loads))
            if new_max < current_max - 1e-12:
                current_bits = new_bits
                current_loads = new_loads
                current_max = new_max
                improved = True
    
    # Double-flip local search
    improved = True
    while improved and time.time() < safety_deadline:
        improved = False
        for i in range(n_demands):
            if time.time() >= safety_deadline:
                break
            for j in range(i + 1, n_demands):
                if time.time() >= safety_deadline:
                    break
                # Try flipping bits i and j
                new_bits = current_bits.copy()
                new_bits[i] = 1.0 - new_bits[i]
                new_bits[j] = 1.0 - new_bits[j]
                new_loads = base + new_bits @ deltas
                new_max = float(np.max(new_loads))
                if new_max < current_max - 1e-12:
                    current_bits = new_bits
                    current_loads = new_loads
                    current_max = new_max
                    improved = True
    
    # Check if local search found a better solution
    if current_max < best_max_load - 1e-12:
        best_max_load = current_max
        best_mask = int(np.sum(current_bits * (1 << np.arange(n_demands))))
    
    # Construct result
    routing = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            routing[i] = 1.0  # CCW
        else:
            routing[i] = 0.0  # CW
            
    return routing
