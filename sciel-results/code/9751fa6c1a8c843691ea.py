# EVOLVE-BLOCK-START
import time
import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n=10 nodes with 10 edges.
    15 demands, each routed CW (0) or CCW (1). Minimize max edge load.
    
    Uses exact brute-force enumeration of all 2^15 routing combinations.
    Includes a deadline guard to ensure termination within 600s.
    """
    start_time = time.time()
    deadline = 600.0
    safety_margin = 10.0 # seconds
    
    n = 10
    D = 15

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
        (5, 2, 2),
    ]

    # Precompute matrices
    cw_mat = np.zeros((D, n), dtype=np.int8)
    ccw_mat = np.zeros((D, n), dtype=np.int8)

    for d, (s, t, w) in enumerate(demands):
        if s <= t:
            cw_edges = range(s, t)
        else:
            cw_edges = list(range(s, n)) + list(range(0, t))
        
        # Set CW edges to 1
        for e in cw_edges:
            cw_mat[d, e] = 1
            
        # CCW is the complement
        ccw_mat[d] = 1 - cw_mat[d]

    weights = np.array([w for (_, _, w) in demands], dtype=np.float64)

    # Base load: all demands routed CW
    base_load = cw_mat.T @ weights  # shape (n,)
    
    # Difference matrix: ccw - cw, values in {-1, 0, 1}
    diff_mat = ccw_mat.astype(np.float64) - cw_mat.astype(np.float64)
    
    # Weighted difference: weight[d] * diff_mat[d]
    weighted_diff = diff_mat * weights[:, None]  # shape (D, n)

    # Enumerate all 2^15 combinations
    # We can do this in chunks if necessary, but 32768 is small.
    num_combos = 1 << D
    
    # Initialize best tracking
    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=np.float64)
    
    # Process in batches to allow for deadline checks if D were larger, 
    # but here we do it all at once as it's fast.
    # However, to strictly follow "deadline guard", we check time.
    
    if time.time() > deadline - safety_margin:
        # Not enough time, return all CW (or current best)
        return best_routing if best_max_load < float('inf') else np.zeros(D, dtype=np.float64)

    # Generate bitmasks
    # Using arange and bit shifting is efficient
    bits = np.arange(num_combos, dtype=np.uint32)
    
    # Compute loads for all combinations
    # loads[i, e] = base_load[e] + sum_d bits[i, d] * weighted_diff[d, e]
    
    # We can compute this without explicitly building the full (32768, 15) float matrix 
    # if we want to save memory, but it's small.
    # Let's stick to the vectorized approach for speed.
    
    # Construct x_mat (num_combos, D)
    x_mat = np.zeros((num_combos, D), dtype=np.float64)
    for d in range(D):
        x_mat[:, d] = ((bits >> d) & 1).astype(np.float64)

    # Matrix multiply: (num_combos, D) @ (D, n) -> (num_combos, n)
    loads = x_mat @ weighted_diff + base_load[None, :]
    
    max_loads = loads.max(axis=1)
    best_idx = int(np.argmin(max_loads))
    best_max_load = max_loads[best_idx]
    
    # Extract routing
    best_bits = bits[best_idx]
    routing = np.zeros(D, dtype=np.float64)
    for d in range(D):
        routing[d] = float((best_bits >> d) & 1)
        
    return routing

# EVOLVE-BLOCK-END
