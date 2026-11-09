import numpy as np
import time

def ring_routing() -> np.ndarray:
    n_nodes = 10
    n_demands = 15
    
    demands = [
        (0,5,3), (1,7,2), (2,9,4), (3,6,1), (0,4,5),
        (2,8,2), (5,9,3), (1,4,2), (6,0,4), (3,8,1),
        (7,2,3), (4,9,2), (8,1,5), (0,6,1), (5,2,2)
    ]
    
    # Precompute CW and CCW edge sets for each demand
    # CW from s to t: edges s, s+1, ..., t-1 (mod 10)
    # CCW: all other edges
    
    cw_edges = []  # list of lists of edge indices
    ccw_edges = []
    
    for s, t, w in demands:
        if s == t:
            cw_e = list(range(n_nodes))  # full circle? but s==t shouldn't happen here
            ccw_e = []
        else:
            cw_e = []
            e = s
            while e != t:
                cw_e.append(e)
                e = (e + 1) % n_nodes
            ccw_e = [e for e in range(n_nodes) if e not in cw_e]
        cw_edges.append(cw_e)
        ccw_edges.append(ccw_e)
    
    # Precompute weight array
    weights = np.array([d[2] for d in demands], dtype=float)
    
    # For each demand, precompute the edge contribution vectors
    # cw_contrib[i] is a vector of length 10, with weight[i] on CW edges, 0 elsewhere
    # ccw_contrib[i] is a vector of length 10, with weight[i] on CCW edges, 0 elsewhere
    
    cw_contrib = np.zeros((n_demands, n_nodes))
    ccw_contrib = np.zeros((n_demands, n_nodes))
    
    for i in range(n_demands):
        for e in cw_edges[i]:
            cw_contrib[i, e] = weights[i]
        for e in ccw_edges[i]:
            ccw_contrib[i, e] = weights[i]
    
    # Now iterate over all 2^15 combinations
    # routing[i] = 0 means CW, 1 means CCW
    # For a given routing, load = sum over i of (cw_contrib[i] if routing[i]==0 else ccw_contrib[i])
    
    best_max_load = float('inf')
    best_routing = np.zeros(n_demands)
    
    start_time = time.time()
    deadline = 620  # safety margin
    
    total = 1 << n_demands  # 32768
    
    for mask in range(total):
        if time.time() > deadline:
            break
        
        loads = np.zeros(n_nodes)
        for i in range(n_demands):
            if (mask >> i) & 1:
                loads += ccw_contrib[i]
            else:
                loads += cw_contrib[i]
        
        max_load = np.max(loads)
        if max_load < best_max_load:
            best_max_load = max_load
            best_routing = np.array([(mask >> i) & 1 for i in range(n_demands)], dtype=float)
    
    return best_routing
