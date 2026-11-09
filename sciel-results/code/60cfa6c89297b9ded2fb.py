# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
        0 -> clockwise, 1 -> counter-clockwise
    """
    n = 10
    D = 15
    
    # Demands (source, target, weight) in order
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
    
    # Precompute edge indices for clockwise and counter-clockwise routes
    cw_edges = []
    ccw_edges = []
    
    for (s, t, w) in demands:
        cw = []
        if s <= t:
            for e in range(s, t):
                cw.append(e)
        else:
            for e in range(s, n):
                cw.append(e)
            for e in range(0, t):
                cw.append(e)
        cw_edges.append(cw)
        
        all_edges = list(range(n))
        ccw = [e for e in all_edges if e not in cw]
        ccw_edges.append(ccw)
    
    weights = np.array([d[2] for d in demands], dtype=np.float64)
    
    # Precompute for each demand, the change in edge loads if we flip from CW to CCW
    # For demand d, flipping changes load on edges in cw_edges[d] by -w and on ccw_edges[d] by +w
    # We can precompute a matrix: flip_effect[d, e] = change in edge e load when demand d is flipped
    # Actually, let's think of it differently.
    # Let's precompute for each demand, the vector of edge loads contributed if CW, and if CCW.
    
    # Precompute contribution vectors
    cw_contrib = np.zeros((D, n), dtype=np.float64)
    ccw_contrib = np.zeros((D, n), dtype=np.float64)
    
    for d in range(D):
        w = weights[d]
        for e in cw_edges[d]:
            cw_contrib[d, e] = w
        for e in ccw_edges[d]:
            ccw_contrib[d, e] = w
    
    # Initial solution: all clockwise
    routing = np.zeros(D, dtype=np.float64)
    current_loads = np.sum(cw_contrib, axis=0)
    current_max = np.max(current_loads)
    best_routing = routing.copy()
    best_max_load = current_max
    
    start_time = time.time()
    deadline = 600.0
    safety_margin = 30.0
    
    # Local search: hill climbing with random restarts
    # At each step, try flipping one demand at a time, keep the best improvement
    # Repeat until no improvement or time runs out
    # Then do random restarts
    
    def evaluate(routing_vec):
        """Compute edge loads and max load for a given routing vector (0=CW, 1=CCW)."""
        loads = np.zeros(n, dtype=np.float64)
        for d in range(D):
            if routing_vec[d] >= 0.5:
                loads += ccw_contrib[d]
            else:
                loads += cw_contrib[d]
        return loads, np.max(loads)
    
    def local_search(routing_vec, time_limit):
        """Perform hill climbing local search starting from routing_vec."""
        nonlocal best_routing, best_max_load
        
        current = routing_vec.copy()
        current_loads, current_max = evaluate(current)
        
        improved = True
        while improved and (time.time() < time_limit):
            improved = False
            best_flip = -1
            best_flip_max = current_max
            best_flip_loads = None
            
            for d in range(D):
                # Try flipping demand d
                new_routing = current.copy()
                new_routing[d] = 1.0 - new_routing[d]
                
                # Compute new loads
                if new_routing[d] >= 0.5:
                    # Switching from CW to CCW
                    new_loads = current_loads - cw_contrib[d] + ccw_contrib[d]
                else:
                    # Switching from CCW to CW
                    new_loads = current_loads - ccw_contrib[d] + cw_contrib[d]
                
                new_max = np.max(new_loads)
                
                if new_max < best_flip_max - 1e-12:
                    best_flip_max = new_max
                    best_flip = d
                    best_flip_loads = new_loads
            
            if best_flip >= 0:
                current[best_flip] = 1.0 - current[best_flip]
                current_loads = best_flip_loads
                current_max = best_flip_max
                improved = True
                
                if current_max < best_max_load - 1e-12:
                    best_max_load = current_max
                    best_routing = current.copy()
        
        return current, current_max
    
    # Multiple random restarts
    np.random.seed(42)
    
    # First, do local search from the all-CW solution
    if time.time() < deadline - safety_margin:
        local_search(routing, deadline - safety_margin)
    
    # Then do random restarts
    max_restarts = 500
    restart_count = 0
    while restart_count < max_restarts and time.time() < deadline - safety_margin:
        # Random initial solution
        random_routing = np.random.rand(D)
        local_search(random_routing, deadline - safety_margin)
        restart_count += 1
    
    return best_routing

# EVOLVE-BLOCK-END
