# EVOLVE-BLOCK-START
import numpy as np
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using exhaustive search with optimized
    computation via precomputed per-edge demand contribution arrays.
    
    Returns:
        routing: np.ndarray of shape (15,) with one entry per demand.
    """
    start_time = time.time()
    DEADLINE = 625.0

    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute CW and CCW contributions as NumPy arrays for fast vector updates
    cw_contrib = np.zeros((D, N), dtype=np.float64)
    ccw_contrib = np.zeros((D, N), dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        cw_mask = 0
        if s <= t:
            for e in range(s, t):
                cw_mask |= (1 << e)
        else:
            for e in range(s, N):
                cw_mask |= (1 << e)
            for e in range(0, t):
                cw_mask |= (1 << e)
        
        all_edges_mask = (1 << N) - 1
        ccw_mask = all_edges_mask ^ cw_mask
        
        for e in range(N):
            if cw_mask & (1 << e):
                cw_contrib[i, e] = w
            if ccw_mask & (1 << e):
                ccw_contrib[i, e] = w

    # Exhaustive search with Gray code
    # Start with all CW (routing 0)
    loads = np.sum(cw_contrib, axis=0)
    best_max_load = np.max(loads)
    best_routing = np.zeros(D, dtype=np.int8)
    
    current_routing = np.zeros(D, dtype=np.int8)
    
    for step in range(1, 1 << D):
        gray = step ^ (step >> 1)
        prev_gray = (step - 1) ^ ((step - 1) >> 1)
        diff = gray ^ prev_gray
        i = diff.bit_length() - 1
        
        old_bit = current_routing[i]
        new_bit = 1 - old_bit
        current_routing[i] = new_bit
        
        if new_bit == 1:
            # Switch from CW to CCW
            loads -= cw_contrib[i]
            loads += ccw_contrib[i]
        else:
            # Switch from CCW to CW
            loads -= ccw_contrib[i]
            loads += cw_contrib[i]
        
        max_load = np.max(loads)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = current_routing.copy()
        
        if step % 4096 == 0 and time.time() > DEADLINE:
            break

    # Return as numpy array of floats
    return best_routing.astype(np.float64)

# EVOLVE-BLOCK-END
