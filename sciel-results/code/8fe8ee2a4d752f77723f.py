import numpy as np

def ring_routing() -> np.ndarray:
    n_nodes = 10
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1),
        (0, 4, 5), (2, 8, 2), (5, 9, 3), (1, 4, 2),
        (6, 0, 4), (3, 8, 1), (7, 2, 3), (4, 9, 2),
        (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    D = len(demands)
    
    # Precompute paths
    cw_paths = []
    ccw_paths = []
    for s, t, w in demands:
        cw = []
        e = s
        while e != t:
            cw.append(e)
            e = (e + 1) % n_nodes
        cw_paths.append(cw)
        
        ccw = [e for e in range(n_nodes) if e not in cw]
        ccw_paths.append(ccw)
        
    weights = np.array([w for _, _, w in demands], dtype=np.float64)
    
    def calc_loads(routing):
        loads = np.zeros(n_nodes)
        for i in range(D):
            if routing[i] < 0.5: # CW
                for e in cw_paths[i]:
                    loads[e] += weights[i]
            else: # CCW
                for e in ccw_paths[i]:
                    loads[e] += weights[i]
        return loads
    
    best_max_load = np.inf
    best_routing = None
    
    # Local Search with Random Restarts
    import random
    for restart in range(50):
        # Random initialization
        routing = np.random.rand(D)
        
        # Hill Climbing
        improved = True
        while improved:
            improved = False
            current_loads = calc_loads(routing)
            current_max = np.max(current_loads)
            
            for i in range(D):
                # Try flipping bit i
                old_dir = routing[i]
                routing[i] = 1.0 - old_dir
                
                new_loads = calc_loads(routing)
                new_max = np.max(new_loads)
                
                if new_max < current_max - 1e-9:
                    current_loads = new_loads
                    current_max = new_max
                    improved = True
                else:
                    routing[i] = old_dir
        
        if current_max < best_max_load:
            best_max_load = current_max
            best_routing = routing.copy()
            
    return best_routing
