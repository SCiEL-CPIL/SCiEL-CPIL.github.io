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
    
    def compute_max_load(routing):
        """Compute maximum edge load for a given routing vector (0=cw, 1=ccw)."""
        loads = np.zeros(n)
        for i in range(D):
            if routing[i] < 0.5:
                # Clockwise
                for e in cw_edges[i]:
                    loads[e] += weights[i]
            else:
                # Counter-clockwise
                for e in ccw_edges[i]:
                    loads[e] += weights[i]
        return np.max(loads)
    
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s limit
    
    # Start with all clockwise
    best_routing = np.zeros(D)
    best_load = compute_max_load(best_routing)
    
    # Local search: try flipping each demand and accept if it improves max load
    improved = True
    while improved:
        improved = False
        for i in range(D):
            # Try flipping demand i
            old_val = best_routing[i]
            best_routing[i] = 1.0 - old_val
            new_load = compute_max_load(best_routing)
            
            if new_load < best_load - 1e-9:
                best_load = new_load
                improved = True
            else:
                # Revert
                best_routing[i] = old_val
    
    # Simulated annealing for further improvement
    T = 100.0
    T_min = 0.1
    cooling = 0.999
    
    current_routing = best_routing.copy()
    current_load = best_load
    best_sa_routing = best_routing.copy()
    best_sa_load = best_load
    
    while T > T_min:
        if time.time() > deadline:
            break
        
        # Pick a random demand to flip
        i = np.random.randint(D)
        old_val = current_routing[i]
        current_routing[i] = 1.0 - old_val
        new_load = compute_max_load(current_routing)
        
        delta = new_load - current_load
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_load = new_load
            if new_load < best_sa_load - 1e-9:
                best_sa_load = new_load
                best_sa_routing = current_routing.copy()
        else:
            current_routing[i] = old_val
        
        T *= cooling
    
    # Use the best found
    if best_sa_load < best_load - 1e-9:
        best_routing = best_sa_routing
        best_load = best_sa_load
    
    # Final local search polish
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for i in range(D):
            old_val = best_routing[i]
            best_routing[i] = 1.0 - old_val
            new_load = compute_max_load(best_routing)
            if new_load < best_load - 1e-9:
                best_load = new_load
                improved = True
            else:
                best_routing[i] = old_val
    
    return best_routing

# EVOLVE-BLOCK-END
