# EVOLVE-BLOCK-START
import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using exhaustive search.
    Since there are only 15 demands, there are 2^15 = 32768 possible routings.
    We evaluate all of them to find the one with the minimum maximum edge load.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)

    # Precompute weight vectors for CW and CCW routing for each demand
    # cw_vectors[i] is an array of length 10, where entry e is w_i if edge e is on CW path, else 0
    cw_vectors = np.zeros((D, n))
    ccw_vectors = np.zeros((D, n))
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise path: s -> s+1 -> ... -> t
        e = s
        while e != t:
            cw_vectors[i, e] = w
            e = (e + 1) % n
        
        # Counter-Clockwise path: all other edges
        # Initialize with weight w for all edges, then zero out CW edges
        ccw_vectors[i, :] = w
        # Zero out the edges that are part of the CW path
        ccw_vectors[i, cw_vectors[i] > 0] = 0

    best_max_load = float('inf')
    best_routing = np.zeros(D)

    # Iterate through all 2^D combinations
    for mask in range(1 << D):
        # Extract bits for each demand
        bits = np.array([(mask >> i) & 1 for i in range(D)], dtype=bool)
        
        # Select vectors: if bit is 1 (CCW), use ccw_vectors; else use cw_vectors
        selected_vectors = np.where(bits[:, None], ccw_vectors, cw_vectors)
        loads = selected_vectors.sum(axis=0)
        current_max = loads.max()
        
        if current_max < best_max_load:
            best_max_load = current_max
            best_routing = bits.astype(float)

    return best_routing

# EVOLVE-BLOCK-END
