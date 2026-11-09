import numpy as np
import time

def ring_routing() -> np.ndarray:
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
    
    cw_matrix = np.zeros((D, n_nodes))
    ccw_matrix = np.zeros((D, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        if s <= t:
            cw_edges = list(range(s, t))
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
        
        for e in cw_edges:
            cw_matrix[i, e] = w
            
        all_edges = set(range(n_nodes))
        cw_set = set(cw_edges)
        ccw_list = [e for e in range(n_nodes) if e not in cw_set]
        for e in ccw_list:
            ccw_matrix[i, e] = w
            
    base_loads = cw_matrix.sum(axis=0)
    delta_matrix = ccw_matrix - cw_matrix
    
    best_load = float('inf')
    best_routing = np.zeros(D)
    
    start_time = time.time()
    deadline = start_time + 600.0
    
    total_bits = 2 ** D
    batch_size = 1024
    
    for batch_start in range(0, total_bits, batch_size):
        if time.time() > deadline - 10:
            break
        
        batch_end = min(batch_start + batch_size, total_bits)
        bits_array = np.arange(batch_start, batch_end, dtype=np.int32)
        
        bit_positions = np.arange(D)
        r_batch = ((bits_array[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1).astype(np.float64)
        
        loads_batch = r_batch @ delta_matrix + base_loads[np.newaxis, :]
        
        max_loads = np.max(loads_batch, axis=1)
        
        local_best_idx = np.argmin(max_loads)
        local_best_load = max_loads[local_best_idx]
        
        if local_best_load < best_load:
            best_load = local_best_load
            best_routing = r_batch[local_best_idx].copy()
            
    return best_routing
