import numpy as np

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    
    # Demands: (source, target, weight)
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
    
    # Precompute edge contributions for each demand
    # cw_loads[i][e] = weight if edge e is in CW path for demand i, else 0
    # ccw_loads[i][e] = weight if edge e is in CCW path for demand i, else 0
    cw_loads = np.zeros((D, N))
    ccw_loads = np.zeros((D, N))
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise path: s -> s+1 -> ... -> t
        # Edges are e, e+1, ..., t-1 mod N
        # If s < t, edges are s, ..., t-1
        # If s >= t, edges are s, ..., N-1 and 0, ..., t-1
        # Simpler: iterate from s to t-1 mod N
        curr = s
        while curr != t:
            cw_loads[i, curr] = w
            curr = (curr + 1) % N
        
        # Counter-clockwise is the complement
        # Total ring edges is N.
        # We can just compute the complement or iterate the other way.
        # Let's compute CCW directly to be safe and clear.
        # CCW path: s -> s-1 -> ... -> t
        curr = s
        while curr != t:
            ccw_loads[i, curr] = w
            curr = (curr - 1) % N
            
    best_max_load = float('inf')
    best_mask = 0
    
    # Exhaustive search over 2^15 combinations
    for mask in range(1 << D):
        # Calculate loads for this mask
        # load[e] = sum(cw_loads[i, e] if bit i is 0 else ccw_loads[i, e])
        # We can vectorize this slightly or just loop.
        # Given D=15, N=10, a simple loop is fine.
        
        # Optimization: We can maintain running loads? No, independent masks.
        # Let's just sum them up.
        loads = np.zeros(N)
        for i in range(D):
            if (mask >> i) & 1:
                loads += ccw_loads[i]
            else:
                loads += cw_loads[i]
        
        current_max = np.max(loads)
        if current_max < best_max_load:
            best_max_load = current_max
            best_mask = mask
            
    # Construct the result array
    # Task: "An entry >= 0.5 routes the demand counter-clockwise, any other value clockwise."
    # So 0.0 for clockwise, 1.0 for counter-clockwise.
    routing = np.zeros(D)
    for i in range(D):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
