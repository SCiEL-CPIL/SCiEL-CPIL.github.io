# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem: 10-node ring, 15 demands,
    each routed CW or CCW to minimize max edge load.
    Uses complete brute-force search over all 2^15 = 32768 routing
    combinations to guarantee the globally optimal solution.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]
    D = len(demands)

    # Precompute edge usage matrices
    A_cw = np.zeros((D, n), dtype=float)
    A_ccw = np.zeros((D, n), dtype=float)
    weights = np.zeros(D, dtype=float)

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        e = s
        while e != t:
            A_cw[d, e % n] = 1
            e = (e + 1) % n
        A_ccw[d] = 1.0 - A_cw[d]

    W_cw = A_cw * weights[:, None]
    W_ccw = A_ccw * weights[:, None]

    # Base load when all demands go CW
    base_loads = W_cw.sum(axis=0)  # shape (n,)
    # Delta load for each demand if it goes CCW instead of CW
    delta = W_ccw - W_cw  # shape (D, n)

    # Brute-force all 2^15 combinations
    best_max_load = float('inf')
    best_routing = np.zeros(D, dtype=float)

    for mask in range(1 << D):
        # Build routing vector: 1.0 for CCW, 0.0 for CW
        routing = np.array([(mask >> d) & 1 for d in range(D)], dtype=float)
        # Compute loads
        loads = base_loads + delta.T @ routing
        max_load = np.max(loads)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = routing.copy()

    return best_routing

# EVOLVE-BLOCK-END
