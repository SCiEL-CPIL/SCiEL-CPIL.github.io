import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges.
    Uses an exhaustive search over all 2^15 possible routing combinations to find
    the one that minimizes the maximum edge load.
    """
    n_nodes = 10
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
    
    # Precompute CW edge usage for each demand
    # W[i, e] is weight of demand i on edge e if routed CW
    W = np.zeros((D, n_nodes))
    V = np.zeros((D, n_nodes)) # CCW usage
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise edges: s, s+1, ..., t-1 (mod n)
        # Generate edges
        edges = []
        curr = s
        while curr != t:
            edges.append(curr)
            curr = (curr + 1) % n_nodes
        for e in edges:
            W[i, e] = w
        
        # Counter-clockwise edges are the complement
        # Since the ring is a cycle, CCW edges are all edges not in CW edges
        # Note: s != t is guaranteed by problem structure (demands are distinct nodes usually, or even if not, path is empty/full)
        # In this specific problem, s != t for all demands.
        # So V[i, :] = w * (1 - W[i, :]/w) where W[i,:] is w or 0.
        # Simpler: V[i, e] = w if W[i, e] == 0 else 0
        V[i, :] = np.where(W[i, :] == 0, w, 0)
        
    # Base load: all CW
    Base = np.sum(W, axis=0)
    
    # Delta load: change from CW to CCW
    # Delta[i, e] = V[i, e] - W[i, e]
    Delta = V - W
    
    # Precompute bit masks for each demand index
    # We will iterate through all 2^D combinations
    best_max_load = float('inf')
    best_routing = np.zeros(D)
    
    # Optimization: Use a lookup or simple iteration
    # 2^15 = 32768 iterations is very fast.
    # We can vectorize the load calculation for a batch of masks?
    # Or just loop.
    
    # To be safe on time, we can do it in chunks or just simply loop.
    # A simple loop with numpy ops inside is ~1ms per 1000 ops? 
    # 32768 * (small vec ops) should be < 1s.
    
    start_time = time.time()
    deadline = 600.0 # 600s limit, keep margin
    
    # We can speed this up by noting that we only need to check if max load improves.
    # But exhaustive search is best.
    
    # Let's use a slightly more optimized loop
    # r = ((mask >> np.arange(D)) & 1)
    # load = Base + Delta.T @ r
    
    for mask in range(1 << D):
        if time.time() > deadline - 10:
            break
        
        # Get routing vector for this mask
        # Bit i is 1 if demand i is CCW
        r = ((mask >> np.arange(D)) & 1).astype(float)
        
        # Calculate load
        # load = Base + sum(r_i * Delta[i])
        # Delta is (D, N). r is (D,).
        # load = Base + Delta.T @ r
        current_load = Base + Delta.T @ r
        
        max_load = np.max(current_load)
        
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = r.copy()
            
    return best_routing
