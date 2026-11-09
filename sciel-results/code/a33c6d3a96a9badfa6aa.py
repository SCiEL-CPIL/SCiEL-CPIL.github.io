# EVOLVE-BLOCK-START
import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    D = 15
    N = 10
    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]

    # Precompute edge masks for each demand for CW and CCW routing
    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        cw_mask = np.zeros(N, dtype=np.int8)
        ccw_mask = np.zeros(N, dtype=np.int8)
        
        # Clockwise from s to t: edges s, s+1, ..., t-1 (mod N)
        steps_cw = (t - s) % N
        if steps_cw > 0:
            for i in range(steps_cw):
                e = (s + i) % N
                cw_mask[e] = 1
        
        # Counter-clockwise from s to t
        steps_ccw = (s - t) % N
        if steps_ccw > 0:
            for i in range(steps_ccw):
                e = (s - 1 - i) % N
                ccw_mask[e] = 1
                
        cw_edges.append(cw_mask)
        ccw_edges.append(ccw_mask)
    
    # Convert to numpy arrays for faster operations
    cw_edges_arr = np.array(cw_edges, dtype=np.int8)  # shape (D, N)
    ccw_edges_arr = np.array(ccw_edges, dtype=np.int8)  # shape (D, N)
    weights = np.array([d[2] for d in demands], dtype=np.float64)  # shape (D,)
    
    best_load = float('inf')
    best_routing = np.zeros(D, dtype=np.float64)
    
    # Time guard
    start_time = time.time()
    deadline = start_time + 500.0
    
    # Iterate over all 2^15 combinations
    for mask in range(1 << D):
        if time.time() > deadline:
            break
            
        # Vectorized load computation
        # For each demand i, if bit i is set, use ccw, else use cw
        bits = ((mask >> np.arange(D)) & 1).astype(np.float64)  # shape (D,)
        
        # Load contribution from CW: where bits==0, use cw_edges_arr[i] * weights[i]
        # Load contribution from CCW: where bits==1, use ccw_edges_arr[i] * weights[i]
        
        # cw_contrib[i, e] = (1 - bits[i]) * cw_edges_arr[i, e] * weights[i]
        # ccw_contrib[i, e] = bits[i] * ccw_edges_arr[i, e] * weights[i]
        # total_load[e] = sum over i of (cw_contrib[i,e] + ccw_contrib[i,e])
        
        # Efficient computation:
        # loads = sum_i [(1-bits[i]) * cw_edges_arr[i] * weights[i] + bits[i] * ccw_edges_arr[i] * weights[i]]
        # loads = sum_i [weights[i] * ((1-bits[i]) * cw_edges_arr[i] + bits[i] * ccw_edges_arr[i])]
        
        # Create selection array: for each demand i, select cw or ccw based on bits
        selected_edges = (1 - bits[:, None]) * cw_edges_arr + bits[:, None] * ccw_edges_arr  # shape (D, N)
        loads = selected_edges.T @ weights  # shape (N,)
        
        current_max_load = np.max(loads)
        
        if current_max_load < best_load:
            best_load = current_max_load
            best_routing = bits.copy()
            
    return best_routing

# EVOLVE-BLOCK-END
