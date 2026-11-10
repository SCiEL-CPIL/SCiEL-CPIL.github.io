import numpy as np
import time

def ring_routing() -> np.ndarray:
    N = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Pre-compute CW and CCW load vectors for each demand
    cw_loads = np.zeros((D, N))
    ccw_loads = np.zeros((D, N))
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise: edges s, s+1, ..., t-1 (mod N)
        # If s == t, no edges? But s != t in this problem.
        # Path: s -> s+1 -> ... -> t
        # Edges involved: s, (s+1)%N, ..., (t-1)%N
        curr = s
        while curr != t:
            cw_loads[i, curr] = w
            curr = (curr + 1) % N
            
        # Counter-clockwise: all other edges
        # Total edges on ring is N.
        # The set of edges in CCW is the complement of CW edges.
        # Note: The union of CW and CCW edges is the entire ring.
        # So ccw_loads[i, :] = w - cw_loads[i, :] ?
        # Wait, the load on an edge is added if the demand uses it.
        # If a demand uses CW, it adds w to CW edges.
        # If a demand uses CCW, it adds w to CCW edges.
        # The set of CCW edges is indeed the complement of the CW edges in the ring.
        # So for a specific demand i, if we route CCW, the load vector is w on the complement edges.
        # Let's verify.
        # Ring edges: 0..9.
        # CW path from s to t uses specific edges.
        # CCW path uses the rest.
        # Yes, they are disjoint and cover the whole ring.
        # So ccw_loads[i, :] = w - cw_loads[i, :] is correct?
        # No, cw_loads[i, :] has w on specific edges, 0 elsewhere.
        # ccw_loads[i, :] should have w on the OTHER edges, 0 elsewhere.
        # So ccw_loads[i, :] = w - cw_loads[i, :] is correct.
        ccw_loads[i, :] = w - cw_loads[i, :]

    best_max_load = float('inf')
    best_mask = 0
    
    # Exhaustive search
    # 2^15 = 32768 iterations.
    # Each iteration involves summing D vectors of size N.
    # We can do this with numpy.
    
    for mask in range(1 << D):
        # Determine directions
        # If bit i is 0, use CW. If 1, use CCW.
        # Let's build the load vector.
        # load = sum(cw_loads[i] if (mask & (1<<i)) == 0 else ccw_loads[i] for i in range(D))
        
        # Vectorized approach:
        # Create a boolean array of bits
        # bits = [(mask >> i) & 1 for i in range(D)]
        # This is slow in Python loop.
        
        # Alternative:
        # Pre-compute sum of all CW loads: total_cw = sum(cw_loads, axis=0)
        # Pre-compute difference: diff = ccw_loads - cw_loads
        # Load for mask = total_cw + sum(diff[i] for i where bit i is 1)
        
        # Let's pre-compute total_cw and diff.
        # This optimization is outside the loop.
        
        pass # Placeholder for logic in actual code below

    # Re-doing with optimization
    total_cw = np.sum(cw_loads, axis=0)
    diff = ccw_loads - cw_loads
    
    # We want to minimize max(total_cw + sum_{i in S} diff[i]) where S is set of bits set in mask.
    
    # We can iterate through masks.
    # To speed up, we can use a Gray code or just simple iteration.
    # Simple iteration is fine for 32k.
    
    start_time = time.time()
    timeout = 590 # 10s margin
    
    # Using a simple loop might be slower than expected due to numpy overhead per iteration?
    # 32768 * (numpy ops)
    # Let's try to batch or just run it.
    
    # Actually, let's just run the loop. It's small.
    for mask in range(1 << D):
        # Calculate load
        # load = total_cw.copy()
        # for i in range(D):
        #     if mask & (1 << i):
        #         load += diff[i]
        
        # This inner loop is 15 steps. 32768 * 15 = 491k steps. Very fast.
        
        load = total_cw.copy()
        m = mask
        i = 0
        while m:
            if m & 1:
                load += diff[i]
            m >>= 1
            i += 1
            # Note: this loop only processes set bits? No, it processes all bits up to the highest set bit.
            # If mask is 0, loop doesn't run. load = total_cw. Correct.
            # If mask has bits, it adds diff.
            # Wait, if m becomes 0, we stop. But we need to ensure we don't miss bits if m was 0 initially?
            # If mask=0, m=0, loop doesn't run. load=total_cw. Correct.
            # If mask=1, m=1, i=0. m&1 is 1. load+=diff[0]. m=0. i=1. Loop ends. Correct.
            
        max_load = np.max(load)
        if max_load < best_max_load:
            best_max_load = max_load
            best_mask = mask
            
        if time.time() - start_time > timeout:
            break

    # Convert best_mask to routing array
    routing = np.zeros(D)
    for i in range(D):
        if best_mask & (1 << i):
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing
