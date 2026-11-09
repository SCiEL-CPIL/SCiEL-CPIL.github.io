import numpy as np
import time

def ring_routing() -> np.ndarray:
    # Demands: (source, target, weight)
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
    
    # Precompute load contributions for each demand in CW and CCW
    # cw_loads[i] is a vector of 10 ints
    # ccw_loads[i] is a vector of 10 ints
    
    cw_loads = np.zeros((n_demands, n_nodes), dtype=np.int64)
    ccw_loads = np.zeros((n_demands, n_nodes), dtype=np.int64)
    
    for i, (s, t, w) in enumerate(demands):
        # Determine CW edges
        if s < t:
            cw_edges = range(s, t)
        elif s > t:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        else:
            cw_edges = []
            
        cw_set = set(cw_edges)
        ccw_edges = [e for e in range(n_nodes) if e not in cw_set]
        
        for e in cw_edges:
            cw_loads[i, e] = w
            
        for e in ccw_edges:
            ccw_loads[i, e] = w
            
    # Vectorized search
    # Load(m) = Base + sum_{i: bit_i(m)=1} Delta_i
    # Base = sum(cw_loads)
    # Delta = ccw_loads - cw_loads
    
    base_load = np.sum(cw_loads, axis=0)
    deltas = ccw_loads - cw_loads
    
    # Generate all masks
    # We can do this in chunks if memory is a concern, but 32768 * 15 * 8 bytes is ~4MB, which is fine.
    # M[k, i] = (k >> i) & 1
    
    # Create indices for bits
    # We can use np.arange and bitwise operations
    masks = np.arange(1 << n_demands, dtype=np.uint32)
    
    # Extract bits
    # M will be shape (32768, 15)
    # M[:, i] = (masks >> i) & 1
    
    # We can construct M by iterating over bits or using broadcasting
    # Let's use broadcasting: masks[:, None] >> np.arange(n_demands)
    # This creates a (32768, 15) array.
    
    # To save memory and time, we can compute the loads directly.
    # Loads = M @ deltas + base_load
    
    # Let's try to compute M efficiently.
    # bit_positions = np.arange(n_demands, dtype=np.uint32)
    # M = (masks[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1
    
    # However, creating M explicitly might be slow.
    # Alternative:
    # We can iterate over masks in Python but use NumPy for the sum? No, that's the current slow part.
    # The matrix multiplication M @ deltas is the key.
    
    # Let's construct M.
    bit_positions = np.arange(n_demands, dtype=np.uint32)
    M = (masks[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1
    
    # M is uint8 or similar. deltas is int64.
    # Cast M to float or int64 for multiplication?
    # M @ deltas will work if dtypes are compatible.
    # Let's cast M to int64 to be safe, or just let NumPy handle it.
    
    loads_all = M @ deltas + base_load[np.newaxis, :]
    
    # loads_all is shape (32768, 10)
    max_loads = np.max(loads_all, axis=1)
    
    best_idx = np.argmin(max_loads)
    best_mask = masks[best_idx]
    
    # Construct result array
    # Entry >= 0.5 means CCW (1), < 0.5 means CW (0)
    routing = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
