# EVOLVE-BLOCK-START
import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solve the ring loading problem by exhaustive search (brute force) 
    over all 2^15 possible routing combinations. 
    This guarantees the optimal solution for this small instance size.
    Optimized with NumPy for vectorized load calculation.
    """
    n_nodes = 10
    n_edges = 10
    D = 15

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

    # Precompute edge masks as boolean arrays for NumPy
    # shape (D, 10)
    cw_masks = np.zeros((D, n_edges), dtype=bool)
    ccw_masks = np.zeros((D, n_edges), dtype=bool)
    weights = np.zeros(D)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        cur = s
        while cur != t:
            cw_masks[d, cur] = True
            cur = (cur + 1) % n_nodes
        
        # CCW is the complement
        ccw_masks[d, :] = ~cw_masks[d, :]

    best_max_load = float('inf')
    best_routing = np.zeros(D)

    # Iterate over all 2^15 combinations
    for mask in range(1 << D):
        # Create a boolean vector indicating CCW routing (True) or CW (False)
        # bit d is 1 -> CCW
        routing_bits = np.array([(mask >> d) & 1 for d in range(D)], dtype=bool)
        
        # Select masks: where routing_bits is True, use ccw_masks, else cw_masks
        # We can use np.where
        # selected_masks shape (D, 10)
        selected_masks = np.where(routing_bits[:, None], ccw_masks, cw_masks)
        
        # Calculate loads: sum of weights[d] * selected_masks[d, e] for all d
        # loads[e] = sum_d weights[d] * selected_masks[d, e]
        # This is a matrix multiplication: (10, D) @ (D,) -> (10,)
        # Or simply: (weights[:, None] * selected_masks).sum(axis=0)
        loads = (weights[:, None] * selected_masks).sum(axis=0)
        
        current_max_load = np.max(loads)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_routing = routing_bits.astype(float)

    return best_routing

# EVOLVE-BLOCK-END
