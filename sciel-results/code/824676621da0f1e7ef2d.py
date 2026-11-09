# EVOLVE-BLOCK-START
import numpy as np


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
    n_nodes = 10
    n_demands = 15
    
    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute edge sets for each demand in both directions
    # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod 10)
    # Counter-clockwise uses all other edges
    
    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        # Clockwise edges: s, s+1, ..., t-1 (mod n)
        cw = set()
        edge = s
        while edge != t:
            cw.add(edge)
            edge = (edge + 1) % n_nodes
        cw_edges.append(cw)
        
        # Counter-clockwise: all other edges
        all_edges = set(range(n_nodes))
        ccw = all_edges - cw
        ccw_edges.append(ccw)
    
    # Precompute edge incidence as bitmasks for faster load computation
    # Each edge is a bit position 0..9
    cw_mask = []
    ccw_mask = []
    for i in range(n_demands):
        m_cw = 0
        for e in cw_edges[i]:
            m_cw |= (1 << e)
        cw_mask.append(m_cw)
        m_ccw = 0
        for e in ccw_edges[i]:
            m_ccw |= (1 << e)
        ccw_mask.append(m_ccw)
    
    weights = np.array([d[2] for d in demands], dtype=np.float64)
    
    def compute_max_load(routing):
        """Compute the maximum edge load given a routing vector."""
        loads = np.zeros(n_nodes, dtype=np.float64)
        for i in range(n_demands):
            if routing[i] >= 0.5:
                mask = ccw_mask[i]
            else:
                mask = cw_mask[i]
            w = weights[i]
            for e in range(n_nodes):
                if mask & (1 << e):
                    loads[e] += w
        return np.max(loads)
    
    # Exhaustive search over all 2^15 = 32768 possible routings
    best_routing = None
    best_load = float('inf')
    
    for bits in range(1 << n_demands):
        routing = np.array([(bits >> i) & 1 for i in range(n_demands)], dtype=np.float64)
        load = compute_max_load(routing)
        if load < best_load:
            best_load = load
            best_routing = routing.copy()
    
    return best_routing


# EVOLVE-BLOCK-END
