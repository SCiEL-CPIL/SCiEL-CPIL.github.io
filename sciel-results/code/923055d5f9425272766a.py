# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def ring_routing() -> np.ndarray:
    """
    Solve the ring loading problem on a ring of n = 10 nodes with 10 edges,
    routing 15 demands either clockwise or counter-clockwise to minimize
    the maximum edge load.
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
    cw_masks = []
    for s, t, w in demands:
        mask = np.zeros(n, dtype=np.int64)
        if s != t:
            for i in range((t - s) % n):
                edge_idx = (s + i) % n
                mask[edge_idx] = 1
        cw_masks.append(mask)

    cw_masks = np.array(cw_masks, dtype=np.int64)  # shape (D, n)
    weights = np.array([w for s, t, w in demands], dtype=np.int64)  # shape (D,)
    ccw_masks = 1 - cw_masks  # shape (D, n)

    base_load = cw_masks.T @ weights  # shape (n,)
    diff_masks = ccw_masks - cw_masks  # shape (D, n)
    contrib = weights[:, np.newaxis] * diff_masks  # shape (D, n)

    start_time = time.time()
    deadline = start_time + 600  # Safety margin

    # First, try brute force since 2^15 is small and guarantees global optimum
    best_max_load = np.inf
    best_routing = np.zeros(D, dtype=np.int64)
    
    # Brute force all 2^15 combinations
    for mask in range(1 << D):
        if time.time() > deadline:
            break
        x = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.int64)
        load = base_load + x @ contrib
        max_load = np.max(load)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = x

    # If brute force completed within deadline, return the result
    if time.time() < deadline - 10:
        return best_routing.astype(np.float64)

    # If time is running out, try a quick local search refinement
    # This is unlikely to be needed given the small problem size, but included for safety
    if time.time() < deadline - 5:
        # Try flipping single bits to see if we can improve
        current = best_routing.copy()
        for _ in range(100):  # Limit iterations
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(D):
                trial = current.copy()
                trial[i] = 1 - trial[i]
                load = base_load + trial @ contrib
                max_load = np.max(load)
                if max_load < best_max_load:
                    best_max_load = max_load
                    best_routing = trial
                    current = trial
                    improved = True
            if not improved:
                break

    return best_routing.astype(np.float64)


# EVOLVE-BLOCK-END
