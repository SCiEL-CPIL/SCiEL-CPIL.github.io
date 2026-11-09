import numpy as np
import time

def ring_routing() -> np.ndarray:
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute CW masks
    cw_mask = np.zeros((D, n), dtype=np.int8)
    for d, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            cw_mask[d, e] = 1
            e = (e + 1) % n
            
    weights = np.array([d[2] for d in demands], dtype=float)
    
    # B: Load if all demands are routed CW
    B = (weights[:, None] * cw_mask).sum(axis=0)
    
    # D_vec[d]: Difference in load vector if demand d is switched from CW to CCW
    # If routed CCW, it uses edges NOT in CW path.
    # Load_CCW(d) = w * (1 - cw_mask[d])
    # Load_CW(d) = w * cw_mask[d]
    # Delta = Load_CCW - Load_CW = w * (1 - 2*cw_mask[d])
    ones = np.ones(n, dtype=np.int8)
    D_vec = weights[:, None] * (ones[None, :] - 2 * cw_mask)
    
    # Deadline guard
    start_time = time.time()
    deadline = 600.0
    
    # Vectorized exhaustive search
    # We want to find binary vector x in {0,1}^D that minimizes max(B + x @ D_vec)
    # Total combinations: 2^15 = 32768
    # We can generate all combinations at once or in chunks.
    # 32768 * 15 * 10 operations is very small.
    
    # Generate all binary vectors of length D
    # Using bit operations
    all_bits = np.arange(2**D, dtype=np.uint32)
    
    # Create matrix of bits: shape (2^D, D)
    # bit i of number g is (g >> i) & 1
    # We can do this vectorized
    powers_of_2 = 1 << np.arange(D)
    x_matrix = ((all_bits[:, None] & powers_of_2[None, :]) != 0).astype(float)
    
    # Calculate loads for all combinations
    # loads shape: (2^D, n)
    # loads = B[None, :] + x_matrix @ D_vec
    loads = B[None, :] + x_matrix @ D_vec
    
    # Find max load for each combination
    max_loads = np.max(loads, axis=1)
    
    # Find the combination with minimum max load
    best_idx = np.argmin(max_loads)
    best_max_load = max_loads[best_idx]
    
    # Extract the best routing decision
    best_routing = x_matrix[best_idx]
    
    # Verify time (should be instant)
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback to all CW or something safe, though this shouldn't happen
        return np.zeros(D)
        
    return best_routing
