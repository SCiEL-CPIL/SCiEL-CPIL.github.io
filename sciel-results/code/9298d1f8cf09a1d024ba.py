# EVOLVE-BLOCK-START
import numpy as np

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
    n = 10
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
    
    # Precompute the edges used for each demand in both directions
    # clockwise_edges[i] is a boolean array of length 10 for demand i
    # counter_clockwise_edges[i] is a boolean array of length 10 for demand i
    
    # We can represent edge loads as a vector of length 10.
    # For each demand, we precompute the weight vector for CW and CCW.
    
    cw_vectors = []
    ccw_vectors = []
    
    for s, t, w in demands:
        cw_vec = np.zeros(n)
        ccw_vec = np.zeros(n)
        
        # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
        # If s <= t, edges are s to t-1
        # If s > t, edges are s to n-1 and 0 to t-1
        
        if s <= t:
            for e in range(s, t):
                cw_vec[e] = w
        else:
            for e in range(s, n):
                cw_vec[e] = w
            for e in range(0, t):
                cw_vec[e] = w
        
        # Counter-clockwise is the complement (all other edges)
        # The total ring has n edges.
        # CCW edges are those NOT in CW path.
        ccw_vec = np.where(cw_vec == 0, w, 0)
        
        # Note: This logic assumes that the CW path and CCW path partition the ring edges.
        # Let's verify:
        # CW path length in edges: (t - s) % n. If 0, it's 0? No, s!=t for all demands here?
        # (0,5): 5 edges. (1,7): 6 edges. (2,9): 7 edges.
        # Wait, the problem says "Clockwise from s to t uses edges s, ..., t-1".
        # If s=0, t=5, edges are 0,1,2,3,4. (5 edges).
        # CCW uses "all other ring edges". The other edges are 5,6,7,8,9. (5 edges).
        # So yes, they partition the ring.
        
        cw_vectors.append(cw_vec)
        ccw_vectors.append(ccw_vec)
    
    cw_vectors = np.array(cw_vectors) # shape (15, 10)
    ccw_vectors = np.array(ccw_vectors) # shape (15, 10)
    
    best_max_load = float('inf')
    best_mask = 0
    
    # Exhaustive search over 2^15 possibilities
    # Using numpy for vectorized calculation of loads for a single mask is not directly
    # efficient for the whole loop unless we vectorize the mask iteration, which is hard.
    # However, 32768 iterations is small. We can optimize the inner loop.
    
    # Alternative: Iterate masks, compute total load vector.
    # Load = sum( w_i * vec_i ) where vec_i is CW or CCW vector based on bit i.
    
    for mask in range(1 << D):
        # Compute load vector
        # We can do this efficiently by summing the selected vectors.
        # Let's build the load vector.
        loads = np.zeros(n)
        
        for i in range(D):
            if (mask >> i) & 1:
                loads += ccw_vectors[i]
            else:
                loads += cw_vectors[i]
        
        current_max_load = np.max(loads)
        
        if current_max_load < best_max_load:
            best_max_load = current_max_load
            best_mask = mask
            
            # Early termination if we hit the theoretical lower bound?
            # Theoretical lower bound is total_weight / n ? No, that's for uniform distribution.
            # A better lower bound is max( sum of weights of demands that MUST cross a cut / 2 ? )
            # Actually, just finding the exact minimum is fast enough.
            if best_max_load == 7: # Just a guess for early stop, but 7 might be too low.
                # Let's not guess, just run through. 32k iterations is nothing.
                pass

    # Construct the routing array
    routing = np.zeros(D)
    for i in range(D):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing

# EVOLVE-BLOCK-END
