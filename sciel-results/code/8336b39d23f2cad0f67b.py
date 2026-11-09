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
    
    # Pre-compute edge usage for CW and CCW
    # A[i, e] is weight added to edge e if demand i is routed CW.
    # B[i, e] is weight added to edge e if demand i is routed CCW.
    
    A = np.zeros((n_demands, n_nodes))
    B = np.zeros((n_demands, n_nodes))
    
    for i, (s, t, w) in enumerate(demands):
        # CW edges: s, s+1, ..., t-1
        current = s
        while current != t:
            A[i, current] = w
            current = (current + 1) % n_nodes
        
        # CCW edges: all others
        for e in range(n_nodes):
            if A[i, e] == 0:
                B[i, e] = w

    # Base load: all demands routed CW
    base_load = A.sum(axis=0)
    
    # Delta: difference between CCW and CW load contributions
    # delta[i] = B[i] - A[i]
    delta = B - A
    
    # Vectorized Exhaustive Search
    # Number of masks is 2^15 = 32768
    # We can compute all loads at once.
    
    # Generate all masks
    masks = np.arange(1 << n_demands)
    
    # Create bits matrix: shape (n_masks, n_demands)
    # bits[m, i] is 1 if bit i is set in mask m
    bits = (masks[:, None] >> np.arange(n_demands)) & 1
    
    # Compute loads for all masks
    # loads_all[m, e] = base_load[e] + sum(bits[m, i] * delta[i, e] for all i)
    # This is a matrix multiplication: bits @ delta
    # bits: (32768, 15)
    # delta: (15, 10)
    # result: (32768, 10)
    
    loads_all = base_load[None, :] + bits @ delta
    
    # Find the mask with the minimum maximum load
    max_loads = loads_all.max(axis=1)
    best_idx = np.argmin(max_loads)
    best_mask = masks[best_idx]
    
    # Construct result array
    # Entry >= 0.5 for CCW (bit 1), < 0.5 for CW (bit 0)
    result = np.zeros(n_demands)
    for i in range(n_demands):
        if (best_mask >> i) & 1:
            result[i] = 1.0
        else:
            result[i] = 0.0
            
    return result
