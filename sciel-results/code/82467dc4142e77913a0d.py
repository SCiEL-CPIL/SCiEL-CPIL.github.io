# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
        0 -> clockwise, 1 (or >= 0.5) -> counter-clockwise
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
        (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute clockwise edge incidence for each demand
    # cw_edges[i] is a list of edge indices used if demand i goes clockwise
    # ccw_edges[i] is a list of edge indices used if demand i goes counter-clockwise
    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        cw = []
        # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
        if s <= t:
            for e in range(s, t):
                cw.append(e)
        else:
            for e in range(s, n):
                cw.append(e)
            for e in range(0, t):
                cw.append(e)
        cw_edges.append(cw)
        
        # Counter-clockwise uses all other edges
        all_edges = set(range(n))
        ccw = list(all_edges - set(cw))
        ccw_edges.append(ccw)
    
    # Weights
    weights = np.array([w for s, t, w in demands], dtype=float)
    
    # Precompute incidence matrix for efficiency
    # A_cw[i, e] = 1 if demand i uses edge e when routed CW
    # A_ccw[i, e] = 1 if demand i uses edge e when routed CCW
    A_cw = np.zeros((D, n))
    A_ccw = np.zeros((D, n))
    for i in range(D):
        for e in cw_edges[i]:
            A_cw[i, e] = 1
        for e in ccw_edges[i]:
            A_ccw[i, e] = 1
            
    # Load matrix: L = A_cw^T * w_cw + A_ccw^T * w_ccw
    # But w_cw and w_ccw depend on the routing choice.
    # Let x be a binary vector of length D. x[i]=0 for CW, x[i]=1 for CCW.
    # Load on edge e = sum_i ( (1-x[i]) * A_cw[i,e] * w[i] + x[i] * A_ccw[i,e] * w[i] )
    # Load_e = sum_i A_cw[i,e]*w[i] + sum_i x[i] * (A_ccw[i,e] - A_cw[i,e]) * w[i]
    
    # Base load (all CW)
    base_loads = A_cw.T @ weights
    
    # Delta load per demand i if flipped to CCW
    # delta[i, e] = (A_ccw[i, e] - A_cw[i, e]) * weights[i]
    delta_loads = (A_ccw - A_cw) * weights[:, np.newaxis]
    
    # Total load for routing x:
    # L(x) = base_loads + delta_loads.T @ x
    
    best_routing = np.zeros(D)
    best_max_load = np.inf
    
    # Exhaustive search over 2^D combinations
    # D=15, so 32768 iterations. Very fast.
    for mask in range(1 << D):
        # Create routing vector x
        # x[i] = 1 if bit i is set, else 0
        x = np.array([(mask >> i) & 1 for i in range(D)], dtype=float)
        
        # Calculate loads
        loads = base_loads + delta_loads.T @ x
        max_load = np.max(loads)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = x
    
    return best_routing

# EVOLVE-BLOCK-END
