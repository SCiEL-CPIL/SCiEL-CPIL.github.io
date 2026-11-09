import numpy as np
import random

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
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
    
    # Precompute edge lists for CW and CCW for each demand
    cw_edges = []
    ccw_edges = []
    weights = []
    
    for s, t, w in demands:
        cw_e = []
        curr = s
        while curr != t:
            cw_e.append(curr)
            curr = (curr + 1) % N
        cw_edges.append(cw_e)
        
        ccw_e = []
        curr = s
        while curr != t:
            curr = (curr - 1) % N
            ccw_e.append(curr)
        ccw_edges.append(ccw_e)
        weights.append(w)

    def calc_loads(mask):
        loads = [0] * N
        for i in range(D):
            w = weights[i]
            if (mask >> i) & 1:
                # CCW
                for e in ccw_edges[i]:
                    loads[e] += w
            else:
                # CW
                for e in cw_edges[i]:
                    loads[e] += w
        return loads

    def local_search(mask):
        current_mask = mask
        current_loads = calc_loads(current_mask)
        current_max = max(current_loads)
        
        improved = True
        while improved:
            improved = False
            for i in range(D):
                # Try flipping demand i
                new_mask = current_mask ^ (1 << i)
                if new_mask == current_mask:
                    continue
                
                new_loads = calc_loads(new_mask)
                new_max = max(new_loads)
                
                if new_max < current_max:
                    current_mask = new_mask
                    current_loads = new_loads
                    current_max = new_max
                    improved = True
                    # Restart loop? Or just continue?
                    # Continuing might miss other flips that become beneficial.
                    # But for small N, restarting is safer to find local optimum.
                    break
        
        return current_mask, current_max

    best_mask = 0
    best_max_load = float('inf')
    
    # Try a few random starts
    random.seed(42)
    for _ in range(50):
        mask = 0
        for i in range(D):
            if random.random() < 0.5:
                mask |= (1 << i)
        
        m, l = local_search(mask)
        if l < best_max_load:
            best_max_load = l
            best_mask = m
            
    # Also try all CW and all CCW
    m, l = local_search(0)
    if l < best_max_load:
        best_max_load = l
        best_mask = m
        
    m, l = local_search((1 << D) - 1)
    if l < best_max_load:
        best_max_load = l
        best_mask = m

    # Construct result
    routing = np.zeros(D)
    for i in range(D):
        if (best_mask >> i) & 1:
            routing[i] = 1.0 # CCW
        else:
            routing[i] = 0.0 # CW
            
    return routing
