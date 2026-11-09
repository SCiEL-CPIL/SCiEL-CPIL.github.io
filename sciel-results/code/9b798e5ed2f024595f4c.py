# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.
    """
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute clockwise edges for each demand as binary matrix
    # cw_matrix[i, e] = 1 if demand i uses edge e when routed clockwise
    cw_matrix = np.zeros((D, N), dtype=np.int8)
    for i, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            cw_matrix[i, e] = 1
            e = (e + 1) % N
    
    # ccw_matrix[i, e] = 1 if demand i uses edge e when routed counter-clockwise
    ccw_matrix = 1 - cw_matrix
    
    weights = np.array([d[2] for d in demands], dtype=float)
    
    def compute_loads_fast(routing):
        """Compute edge loads using matrix operations. routing: 0=cw, 1=ccw"""
        # routing is binary: 0 or 1
        # If routing[i] = 0, use cw_matrix[i]; if 1, use ccw_matrix[i]
        # loads[e] = sum_i weights[i] * (routing[i] * ccw_matrix[i,e] + (1-routing[i]) * cw_matrix[i,e])
        # = sum_i weights[i] * cw_matrix[i,e] + sum_i weights[i] * routing[i] * (ccw_matrix[i,e] - cw_matrix[i,e])
        base_loads = weights @ cw_matrix  # all clockwise
        delta = ccw_matrix - cw_matrix  # -1, 0, +1
        # For each demand, if routed ccw, add weight * delta[i,e]
        # routing is 0/1
        loads = base_loads + (routing * weights) @ delta
        return loads
    
    def max_load_fast(routing):
        return np.max(compute_loads_fast(routing))
    
    deadline = time.time() + 590  # safety margin
    
    # Brute force: 2^15 = 32768 combinations, very fast
    # This guarantees global optimum
    best_routing = np.zeros(D)
    best_ml = float('inf')
    
    # Use numpy to vectorize over all 2^15 combinations
    # For each combination (bitmask), routing[i] = (bitmask >> i) & 1
    n_combos = 1 << D
    # Process in chunks to avoid memory issues
    chunk_size = 10000
    for start in range(0, n_combos, chunk_size):
        end = min(start + chunk_size, n_combos)
        masks = np.arange(start, end, dtype=np.uint32)
        # Extract bits: routing[i] = (mask >> i) & 1
        # Build routing matrix: (chunk, D)
        routing_chunk = np.zeros((end - start, D), dtype=np.float64)
        for i in range(D):
            routing_chunk[:, i] = (masks >> i) & 1
        
        # Compute loads for all chunks: (chunk, N)
        # loads = base + routing_chunk @ (weights * delta).T ... wait
        # loads[e] = base_loads[e] + sum_i routing[i] * weights[i] * delta[i,e]
        # = base_loads[e] + sum_i routing[i] * (weights * delta)[i,e]
        weighted_delta = weights[:, None] * delta  # (D, N)
        loads_chunk = routing_chunk @ weighted_delta + base_loads[None, :]  # (chunk, N)
        max_loads = np.max(loads_chunk, axis=1)
        
        # Find best in this chunk
        local_best_idx = np.argmin(max_loads)
        local_best_ml = max_loads[local_best_idx]
        
        if local_best_ml < best_ml:
            best_ml = local_best_ml
            best_routing = routing_chunk[local_best_idx].copy()
        
        if time.time() > deadline - 10:
            break
    
    return best_routing


# EVOLVE-BLOCK-END
