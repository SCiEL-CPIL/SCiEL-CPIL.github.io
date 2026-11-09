# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Returns:
        routing: np.ndarray of shape (15,) with one value per demand.
                 >= 0.5 means counter-clockwise, < 0.5 means clockwise.
    """
    # Problem setup
    n = 10  # number of nodes/edges
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute clockwise and counter-clockwise edge sets for each demand
    # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
    # Counter-clockwise: all other edges
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        # Clockwise edges
        cw = set()
        cur = s
        while cur != t:
            cw.add(cur)
            cur = (cur + 1) % n
        cw_edges.append(cw)
        
        # Counter-clockwise edges (complement)
        all_edges = set(range(n))
        ccw = all_edges - cw
        ccw_edges.append(ccw)
    
    # Convert to numpy arrays for fast operations
    # For each demand, which edges does it use if CW? if CCW?
    cw_mask = np.zeros((D, n), dtype=np.int8)
    ccw_mask = np.zeros((D, n), dtype=np.int8)
    weights = np.zeros(D)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        for e in cw_edges[i]:
            cw_mask[i, e] = 1
        for e in ccw_edges[i]:
            ccw_mask[i, e] = 1
    
    def compute_max_load(routing):
        """Compute max edge load given routing decisions (0=CW, 1=CCW)."""
        loads = np.zeros(n)
        for i in range(D):
            if routing[i] >= 0.5:
                loads += weights[i] * ccw_mask[i]
            else:
                loads += weights[i] * cw_mask[i]
        return np.max(loads)
    
    # Phase 1: Brute force all 2^15 = 32768 combinations to find optimal
    start_time = time.time()
    deadline = 620.0  # safety margin
    
    best_routing = np.zeros(D)
    best_max_load = float('inf')
    
    # Brute force
    for bits in range(2**D):
        if time.time() > deadline:
            break
        routing = np.zeros(D)
        for i in range(D):
            routing[i] = (bits >> i) & 1
        
        max_load = compute_max_load(routing)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = routing.copy()
    
    # Phase 2: Local search polish - try flipping individual demands
    improved = True
    while improved:
        improved = False
        for i in range(D):
            if time.time() > deadline:
                break
            trial = best_routing.copy()
            trial[i] = 1.0 - trial[i]  # flip
            max_load = compute_max_load(trial)
            if max_load < best_max_load:
                best_max_load = max_load
                best_routing = trial
                improved = True
    
    # Phase 3: Pair flips for additional improvement
    improved = True
    while improved:
        improved = False
        for i in range(D):
            for j in range(i+1, D):
                if time.time() > deadline:
                    break
                trial = best_routing.copy()
                trial[i] = 1.0 - trial[i]
                trial[j] = 1.0 - trial[j]
                max_load = compute_max_load(trial)
                if max_load < best_max_load:
                    best_max_load = max_load
                    best_routing = trial
                    improved = True
        if time.time() > deadline:
            break
    
    return best_routing


# EVOLVE-BLOCK-END
