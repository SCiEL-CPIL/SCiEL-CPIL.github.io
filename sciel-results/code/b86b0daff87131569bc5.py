import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
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
        (5, 2, 2)
    ]
    
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute edge contributions for CW and CCW
    V_cw = np.zeros((n_demands, n_nodes))
    V_ccw = np.zeros((n_demands, n_nodes))
    
    for d, (s, t, w) in enumerate(demands):
        # Determine CW edges: s, s+1, ..., t-1 (mod n_nodes)
        if s < t:
            cw_edges = range(s, t)
        else:
            cw_edges = list(range(s, n_nodes)) + list(range(0, t))
            
        for e in cw_edges:
            V_cw[d, e] = w
            
        for e in range(n_nodes):
            if e not in cw_edges:
                V_ccw[d, e] = w

    # Exhaustive search over all 2^15 = 32768 assignments
    num_assignments = 1 << n_demands
    indices = np.arange(num_assignments, dtype=np.int32)
    
    bit_positions = np.arange(n_demands, dtype=np.int32)
    bits = ((indices[:, None] >> bit_positions) & 1).astype(np.float64)
    
    Diff = V_ccw - V_cw  # Shape (15, 10)
    Base = V_cw.sum(axis=0)  # Shape (10,)
    
    # Compute loads for all assignments: (32768, 15) @ (15, 10) -> (32768, 10)
    Loads = bits @ Diff + Base
    
    # Find max load for each assignment
    MaxLoads = Loads.max(axis=1)
    
    # Find all assignments with the minimum max load
    best_max_load = MaxLoads.min()
    best_indices = np.where(MaxLoads == best_max_load)[0]
    
    # Pick the first one (deterministic)
    best_idx = best_indices[0]
    best_routing = bits[best_idx]
    
    return best_routing
