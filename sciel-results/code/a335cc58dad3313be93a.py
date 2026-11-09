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
        >= 0.5 -> route the demand counter-clockwise
        < 0.5  -> route the demand clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
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
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        if s <= t:
            edges = list(range(s, t))
        else:
            edges = list(range(s, n_nodes)) + list(range(0, t))
        cw_edges.append(edges)
        
        all_edges = set(range(n_nodes))
        cw_set = set(edges)
        ccw_list = [e for e in range(n_nodes) if e not in cw_set]
        ccw_edges.append(ccw_list)
    
    # Precompute load matrices
    cw_matrix = np.zeros((D, n_nodes))
    ccw_matrix = np.zeros((D, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        for e in cw_edges[i]:
            cw_matrix[i, e] = w
        for e in ccw_edges[i]:
            ccw_matrix[i, e] = w
    
    base_loads = cw_matrix.sum(axis=0)
    delta_matrix = ccw_matrix - cw_matrix  # (D, n_nodes)
    delta_matrix_T = delta_matrix.T  # (n_nodes, D)
    
    # Brute force all 2^15 combinations with vectorized bit extraction
    best_load = float('inf')
    best_routing = np.zeros(D)
    
    start_time = time.time()
    deadline = start_time + 600.0
    
    total_bits = 2 ** D
    # Process in batches for efficiency
    batch_size = 1024
    
    for batch_start in range(0, total_bits, batch_size):
        if time.time() > deadline - 10:
            break
        
        batch_end = min(batch_start + batch_size, total_bits)
        bits_array = np.arange(batch_start, batch_end, dtype=np.int32)
        
        # Extract binary vectors for all bits in batch
        # r[i, j] = bit j of bits_array[i]
        bit_positions = np.arange(D)
        r_batch = ((bits_array[:, np.newaxis] >> bit_positions[np.newaxis, :]) & 1).astype(np.float64)
        
        # Compute loads for all routings in batch: (batch_size, D) @ (D, n_nodes) + (n_nodes,)
        loads_batch = r_batch @ delta_matrix + base_loads[np.newaxis, :]
        
        # Find max load for each routing in batch
        max_loads = np.max(loads_batch, axis=1)
        
        # Find the best routing in this batch
        local_best_idx = np.argmin(max_loads)
        local_best_load = max_loads[local_best_idx]
        
        if local_best_load < best_load:
            best_load = local_best_load
            best_routing = r_batch[local_best_idx].copy()
    
    return best_routing

# EVOLVE-BLOCK-END
