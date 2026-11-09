import numpy as np
import time

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    D = 15
    N = 10  # number of nodes/edges
    
    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute matrices
    M_cw = np.zeros((D, N), dtype=np.float64)
    M_ccw = np.zeros((D, N), dtype=np.float64)
    weights = np.zeros(D, dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        # CW from s to t: edges s, s+1, ..., t-1 (mod 10)
        e = s
        while e != t:
            M_cw[i, e] = 1.0
            e = (e + 1) % N
        # CCW from s to t: edges t, t+1, ..., s-1 (mod 10)
        e = t
        while e != s:
            M_ccw[i, e] = 1.0
            e = (e + 1) % N
            
    # Linearize the load calculation
    # Load_e = sum_i w_i * ( (1-x_i)*M_cw[i,e] + x_i*M_ccw[i,e] )
    # Load_e = sum_i w_i * M_cw[i,e] + sum_i x_i * w_i * (M_ccw[i,e] - M_cw[i,e])
    # Load_e = C_e + sum_i x_i * B[i,e]
    
    B = weights[:, np.newaxis] * (M_ccw - M_cw)
    C = np.sum(weights[:, np.newaxis] * M_cw, axis=0)
    
    # Exhaustive search over all 2^D routings using vectorization
    # 2^15 = 32768 combinations
    n_routings = 1 << D
    indices = np.arange(n_routings, dtype=np.uint32)
    
    # Create X matrix of shape (n_routings, D)
    # X[k, i] is 1 if demand i is routed CCW in configuration k
    X = np.zeros((n_routings, D), dtype=np.float64)
    for i in range(D):
        X[:, i] = (indices >> i) & 1
        
    # Compute all loads at once
    # L shape: (n_routings, N)
    L = C[np.newaxis, :] + X @ B
    
    # Find the configuration with the minimum maximum edge load
    max_loads = np.max(L, axis=1)
    best_idx = np.argmin(max_loads)
    
    # Extract the best routing vector
    # best_idx is an integer representing the bitmask
    best_routing_bits = (indices[best_idx] >> np.arange(D)) & 1
    routing = best_routing_bits.astype(np.float64)
    
    return routing
