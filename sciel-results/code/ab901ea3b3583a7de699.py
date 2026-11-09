import numpy as np
import time

def ring_routing() -> np.ndarray:
    # Data
    demands = [
        (0,5,3), (1,7,2), (2,9,4), (3,6,1), (0,4,5), 
        (2,8,2), (5,9,3), (1,4,2), (6,0,4), (3,8,1), 
        (7,2,3), (4,9,2), (8,1,5), (0,6,1), (5,2,2)
    ]
    n_nodes = 10
    n_demands = len(demands)
    
    # Precompute edges
    cw_edges = []
    ccw_edges = []
    for s, t, w in demands:
        cw = set()
        ccw = set()
        # CW: s, s+1, ..., t-1
        cur = s
        while cur != t:
            cw.add(cur)
            cur = (cur + 1) % n_nodes
        # CCW: all others
        all_edges = set(range(n_nodes))
        ccw = all_edges - cw
        cw_edges.append(cw)
        ccw_edges.append(ccw)
        
    # Convert to lists for faster access or numpy arrays
    # Let's use numpy arrays of shape (15, 10) with 0/1 indicating edge usage
    cw_mask = np.zeros((n_demands, n_nodes))
    ccw_mask = np.zeros((n_demands, n_nodes))
    weights = np.array([d[2] for d in demands])
    
    for i, (s, t, w) in enumerate(demands):
        cur = s
        while cur != t:
            cw_mask[i, cur] = 1
            cur = (cur + 1) % n_nodes
        ccw_mask[i] = 1 - cw_mask[i]
        
    def get_max_load(routing):
        # routing: array of 0/1 (0=CW, 1=CCW)
        # load = sum over demands of weight * (routing? ccw_mask : cw_mask)
        # We can compute this as:
        # load_e = sum_i w_i * (routing_i * ccw_mask[i,e] + (1-routing_i) * cw_mask[i,e])
        # load_e = sum_i w_i * cw_mask[i,e] + sum_i w_i * routing_i * (ccw_mask[i,e] - cw_mask[i,e])
        
        # Base load (all CW)
        # delta load per demand if CCW: w_i * (ccw_mask[i] - cw_mask[i])
        
        # Let's just do the direct sum for simplicity and speed check
        # routing is 0/1
        # effective_mask = routing[:, None] * ccw_mask + (1 - routing[:, None]) * cw_mask
        # loads = effective_mask.T @ weights
        # return np.max(loads)
        
        # This is O(15*10) which is fine.
        # But for local search, incremental update is better.
        pass
        
    # Local Search Implementation
    def local_search(initial_routing):
        current_routing = initial_routing.copy()
        current_loads = np.sum(weights[:, None] * (current_routing[:, None] * ccw_mask + (1 - current_routing[:, None]) * cw_mask), axis=0)
        current_max = np.max(current_loads)
        
        improved = True
        while improved:
            improved = False
            for i in range(n_demands):
                # Try flipping i
                new_routing = current_routing.copy()
                new_routing[i] = 1 - new_routing[i]
                
                # Calculate new loads
                # We can update incrementally
                # If we flip i:
                # old contribution: w_i * (routing_i * ccw + (1-routing_i) * cw)
                # new contribution: w_i * (new_routing_i * ccw + (1-new_routing_i) * cw)
                # diff = w_i * (new_routing_i - routing_i) * (ccw - cw)
                
                # Let's just recalc for simplicity first, it's fast enough
                new_loads = np.sum(weights[:, None] * (new_routing[:, None] * ccw_mask + (1 - new_routing[:, None]) * cw_mask), axis=0)
                new_max = np.max(new_loads)
                
                if new_max < current_max:
                    current_routing = new_routing
                    current_loads = new_loads
                    current_max = new_max
                    improved = True
        return current_routing, current_max

    # Run multiple restarts
    best_routing = None
    best_max = float('inf')
    
    start_time = time.time()
    timeout = 500 # Safety margin
    
    # 1. Greedy Start
    greedy_routing = np.zeros(n_demands)
    # Greedy: start all CW, then flip if it helps
    # Actually, let's just use the local search from all-CW and all-CCW and some randoms
    
    # Start 1: All CW
    r, m = local_search(np.zeros(n_demands))
    if m < best_max:
        best_max = m
        best_routing = r
        
    # Start 2: All CCW
    r, m = local_search(np.ones(n_demands))
    if m < best_max:
        best_max = m
        best_routing = r
        
    # Random Restarts
    np.random.seed(42) # Fixed seed for reproducibility? No, let's use random
    while time.time() - start_time < timeout:
        random_routing = np.random.randint(0, 2, n_demands)
        r, m = local_search(random_routing)
        if m < best_max:
            best_max = m
            best_routing = r
            
    return best_routing
