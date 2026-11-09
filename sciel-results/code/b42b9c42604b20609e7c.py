# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the ring loading problem on a ring of n = 10 nodes with 10 edges,
    routing 15 demands either clockwise or counter-clockwise to minimize
    the maximum edge load.
    
    Since the number of demands (15) is small, we use an exhaustive search
    over all 2^15 possible routing configurations to guarantee the global optimum.
    """
    n = 10
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
    D = len(demands)

    # Precompute clockwise edge masks for each demand.
    # cw_mask[i, j] is 1 if edge j is on the clockwise path of demand i.
    cw_masks = np.zeros((D, n), dtype=np.int64)
    for i, (s, t, w) in enumerate(demands):
        if s != t:
            # Clockwise path from s to t uses edges s, s+1, ..., t-1 (mod n)
            length = (t - s) % n
            for k in range(length):
                edge_idx = (s + k) % n
                cw_masks[i, edge_idx] = 1

    weights = np.array([w for s, t, w in demands], dtype=np.int64)
    
    # Base load if all demands go clockwise
    base_load = cw_masks.T @ weights
    
    # Contribution of switching demand i to CCW
    # If demand i goes CW, load on edge j is w_i * cw_masks[i,j]
    # If demand i goes CCW, load on edge j is w_i * (1 - cw_masks[i,j])
    # Change in load = w_i * (1 - 2*cw_masks[i,j])
    # Let x_i be 1 if CCW, 0 if CW.
    # Load_j = base_load_j + sum_i x_i * w_i * (1 - 2*cw_masks[i,j])
    diff_contrib = weights[:, np.newaxis] * (1 - 2 * cw_masks)
    
    start_time = time.time()
    deadline = start_time + 600
    
    best_max_load = np.inf
    best_routing = np.zeros(D, dtype=np.int64)
    
    # Theoretical lower bound for max load
    total_weight = np.sum(weights)
    lower_bound = total_weight / n
    
    # Exhaustive search
    for mask in range(1 << D):
        if time.time() > deadline:
            break
            
        # Create binary vector x from mask
        # x_i = 1 if bit i is set (CCW), else 0 (CW)
        x = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.int64)
        
        # Calculate load
        load = base_load + x @ diff_contrib
        max_load = np.max(load)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = x.copy()
            
            # Early termination if we reach the theoretical lower bound
            if best_max_load <= lower_bound + 1e-9:
                break

    return best_routing.astype(np.float64)

# EVOLVE-BLOCK-END
