import numpy as np

def ring_routing() -> np.ndarray:
    n_nodes = 10
    n_demands = 15
    
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
    
    # Pre-calculate A_cw and A_ccw
    # A_cw[i, e] = 1 if demand i uses edge e when routed CW
    A_cw = np.zeros((n_demands, n_nodes), dtype=np.float64)
    A_ccw = np.zeros((n_demands, n_nodes), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        # Determine CW edges
        # Path s -> t
        # Edges are s, s+1, ..., t-1 (mod 10)
        current = s
        while current != t:
            A_cw[i, current] = 1.0
            current = (current + 1) % n_nodes
        
        # CCW edges are the complement
        A_ccw[i] = 1.0 - A_cw[i]
        
    # Weights
    weights = np.array([d[2] for d in demands], dtype=np.float64)
    
    # Base load if all routed CW
    # L_base[e] = sum_i w_i * A_cw[i, e]
    L_base = A_cw.T @ weights
    
    # Difference matrix D
    # D[i, e] = w_i * (A_ccw[i, e] - A_cw[i, e])
    # If we switch demand i from CW to CCW, the change in load on edge e is D[i, e]
    D = (A_ccw - A_cw) * weights[:, np.newaxis]
    
    # Generate all 2^15 masks
    # M is shape (2^15, 15)
    # M[k, i] is 1 if demand i is CCW in mask k
    # We can use np.arange(2**15) and bit operations
    max_masks = 2 ** n_demands
    # This might be memory heavy if not careful, but 32768 * 15 * 8 bytes ~ 4MB. Fine.
    
    # Efficient way to generate binary matrix
    # indices = np.arange(max_masks)
    # M = ((indices[:, None] >> np.arange(n_demands)[None, :]) & 1).astype(np.float64)
    
    # Alternatively, loop might be faster in Python for such small N if NumPy overhead is high?
    # No, vectorized is better.
    
    indices = np.arange(max_masks)
    M = ((indices[:, None] >> np.arange(n_demands)[None, :]) & 1).astype(np.float64)
    
    # Calculate all loads
    # L_all = L_base + M @ D
    # L_base is (10,), M is (32768, 15), D is (15, 10)
    # M @ D is (32768, 10)
    L_all = L_base[None, :] + M @ D
    
    # Find max load for each mask
    max_loads = L_all.max(axis=1)
    
    # Find the mask with the minimum max load
    best_idx = np.argmin(max_loads)
    best_mask = M[best_idx]
    
    # best_mask is 1 for CCW, 0 for CW
    # The task requires: >= 0.5 routes CCW, < 0.5 routes CW
    # So we can return best_mask directly as it contains 0.0 and 1.0
    return best_mask
