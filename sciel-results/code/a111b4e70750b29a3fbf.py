# EVOLVE-BLOCK-START
import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solve the ring loading problem by exhaustive search (brute force) 
    over all 2^15 possible routing combinations. 
    This guarantees the optimal solution for this small instance size.
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

    # Precompute edge masks for CW and CCW routing for each demand
    # mask is a list of 10 booleans or an integer bitmask
    cw_masks = [0] * D
    ccw_masks = [0] * D
    weights = [0] * D

    for d, (s, t, w) in enumerate(demands):
        weights[d] = w
        cw_set = set()
        cur = s
        while cur != t:
            cw_set.add(cur)
            cur = (cur + 1) % n_nodes
        
        # Create bitmasks
        m_cw = 0
        m_ccw = 0
        for e in range(n_edges):
            if e in cw_set:
                m_cw |= (1 << e)
            else:
                m_ccw |= (1 << e)
        
        cw_masks[d] = m_cw
        ccw_masks[d] = m_ccw

    best_max_load = float('inf')
    best_routing = np.zeros(D)

    # Iterate over all 2^15 combinations
    for mask in range(1 << D):
        # Calculate loads for each edge
        # We can do this efficiently by accumulating weights per edge
        loads = [0.0] * n_edges
        
        for d in range(D):
            w = weights[d]
            if (mask >> d) & 1:
                # Routed CCW
                m = ccw_masks[d]
                for e in range(n_edges):
                    if (m >> e) & 1:
                        loads[e] += w
            else:
                # Routed CW
                m = cw_masks[d]
                for e in range(n_edges):
                    if (m >> e) & 1:
                        loads[e] += w
        
        current_max_load = max(loads)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            # Extract routing vector
            routing = np.zeros(D)
            for d in range(D):
                if (mask >> d) & 1:
                    routing[d] = 1.0
            best_routing = routing

    return best_routing

# EVOLVE-BLOCK-END
