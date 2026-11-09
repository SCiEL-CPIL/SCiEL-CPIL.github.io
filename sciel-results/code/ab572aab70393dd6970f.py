import numpy as np

def ring_routing() -> np.ndarray:
    n_nodes = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    n_demands = len(demands)
    n_edges = n_nodes
    
    # L0: load if all demands are routed CW
    # C: change in load if demand d is switched from CW to CCW
    # L0[e] = sum(w_d for d if e in CW_d)
    # C[d, e] = w_d if e in CCW_d else -w_d
    
    L0 = np.zeros(n_edges)
    C = np.zeros((n_demands, n_edges))
    
    for i, (s, t, w) in enumerate(demands):
        # Determine CW edges
        cw_edges = set()
        curr = s
        while curr != t:
            cw_edges.add(curr)
            curr = (curr + 1) % n_nodes
        
        for e in range(n_edges):
            if e in cw_edges:
                L0[e] += w
                C[i, e] = -w
            else:
                C[i, e] = w
                
    # Generate all 2^15 combinations of routing decisions
    # x_d = 0 -> CW, x_d = 1 -> CCW
    # Load_e = L0[e] + sum(x_d * C[d, e])
    
    # Use itertools.product or numpy bit manipulation
    # 2^15 is small, so we can just iterate or use vectorized approach
    
    # Vectorized approach:
    # Create array of indices 0 to 2^15 - 1
    # Convert to binary matrix of shape (2^15, 15)
    
    n_combos = 1 << n_demands
    # indices = np.arange(n_combos)
    # bits = ((indices[:, None] >> np.arange(n_demands)) & 1)
    
    # This creates a (32768, 15) matrix
    indices = np.arange(n_combos)
    bits = ((indices[:, None] >> np.arange(n_demands)) & 1).astype(float)
    
    # Calculate loads for all combinations
    # loads shape: (32768, 10)
    loads = L0[None, :] + bits @ C
    
    # Find max load for each combination
    max_loads = loads.max(axis=1)
    
    # Find the combination with minimum max load
    best_idx = np.argmin(max_loads)
    best_bits = bits[best_idx]
    
    # The problem asks for an array where >= 0.5 is CCW, else CW.
    # Our bits are 0 or 1. 1 is CCW.
    return best_bits
