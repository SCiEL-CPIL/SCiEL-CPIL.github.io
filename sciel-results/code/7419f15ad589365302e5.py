# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


def _compute_loads(routing, demands, n=10):
    """Compute edge loads given routing decisions."""
    loads = np.zeros(n)
    for i, (s, t, w) in enumerate(demands):
        if routing[i] >= 0.5:
            # counter-clockwise: edges from t to s going backwards
            # i.e., edges (s-1)%n, (s-2)%n, ..., t%n
            # Actually, let's think carefully.
            # Clockwise from s to t uses edges s, s+1, ..., t-1 (mod n)
            # Counter-clockwise uses all OTHER edges.
            # The ring has edges 0..9 where edge e joins node e and (e+1)%n.
            # Clockwise from s to t: if s <= t, edges s, s+1, ..., t-1
            #                    if s > t, edges s, s+1, ..., n-1, 0, ..., t-1
            # Counter-clockwise from s to t: the complement
            if s < t:
                # CCW uses edges t, t+1, ..., n-1, 0, ..., s-1
                if t < n - 1:
                    loads[t:] += w
                if s > 0:
                    loads[:s] += w
            elif s > t:
                # CCW uses edges t, t+1, ..., s-1
                loads[t:s] += w
            # if s == t, no edges (but shouldn't happen)
        else:
            # clockwise
            if s < t:
                loads[s:t] += w
            elif s > t:
                loads[s:] += w
                loads[:t] += w
            # if s == t, no edges
    return loads


def _max_load(routing, demands, n=10):
    loads = _compute_loads(routing, demands, n)
    return np.max(loads)


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    deadline = time.time() + 580  # safety margin
    
    # Precompute which edges each demand uses in each direction
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw = set()
        ccw = set()
        if s < t:
            cw = set(range(s, t))
            ccw = set(range(t, n)) | set(range(0, s))
        elif s > t:
            cw = set(range(s, n)) | set(range(0, t))
            ccw = set(range(t, s))
        else:
            cw = set()
            ccw = set()
        cw_edges.append(cw)
        ccw_edges.append(ccw)
    
    # Start with all clockwise
    best_routing = np.zeros(D)
    best_ml = _max_load(best_routing, demands, n)
    
    # Local search: try flipping each demand
    improved = True
    while improved:
        improved = False
        for i in range(D):
            if time.time() > deadline:
                break
            # Try flipping demand i
            new_routing = best_routing.copy()
            new_routing[i] = 1 - new_routing[i]
            ml = _max_load(new_routing, demands, n)
            if ml < best_ml:
                best_ml = ml
                best_routing = new_routing
                improved = True
        if time.time() > deadline:
            break
    
    # Try some random restarts if time permits
    np.random.seed(42)
    for restart in range(1000):
        if time.time() > deadline:
            break
        # Random initial solution
        routing = np.random.randint(0, 2, D).astype(float)
        ml = _max_load(routing, demands, n)
        
        # Local search from this point
        improved = True
        while improved:
            improved = False
            for i in range(D):
                if time.time() > deadline:
                    break
                new_routing = routing.copy()
                new_routing[i] = 1 - new_routing[i]
                ml_new = _max_load(new_routing, demands, n)
                if ml_new < ml:
                    ml = ml_new
                    routing = new_routing
                    improved = True
            if time.time() > deadline:
                break
        
        if ml < best_ml:
            best_ml = ml
            best_routing = routing
    
    return best_routing


# EVOLVE-BLOCK-END
