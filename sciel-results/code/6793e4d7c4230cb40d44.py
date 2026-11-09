# EVOLVE-BLOCK-START
import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using Multi-start Hill Climbing.
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
    
    # Precompute weight vectors for CW and CCW directions
    cw_vectors = np.zeros((D, n))
    ccw_vectors = np.zeros((D, n))
    
    for i, (s, t, w) in enumerate(demands):
        # Clockwise edges: s, s+1, ..., t-1 (mod n)
        if s < t:
            cw_vectors[i, s:t] = w
        elif s > t:
            cw_vectors[i, s:n] = w
            cw_vectors[i, 0:t] = w
        # If s == t, no edges? But problem implies s != t for routing paths usually.
        # If s == t, load is 0 everywhere? Or 0 length path? 
        # In this specific problem, all s != t.
        
        # Counter-clockwise is the complement
        # We can compute it as total weight vector minus CW vector? 
        # No, the weight is added to edges on the path.
        # The set of edges in CCW is the set of edges NOT in CW.
        # So ccw_vector[i, edge] = w if cw_vector[i, edge] == 0 else 0.
        ccw_vectors[i] = np.where(cw_vectors[i] == 0, w, 0)
    
    def calculate_max_load(mask):
        """Calculate the maximum edge load for a given routing mask."""
        # mask is an integer where bit i is 1 if demand i is CCW, 0 if CW
        loads = np.zeros(n)
        for i in range(D):
            if (mask >> i) & 1:
                loads += ccw_vectors[i]
            else:
                loads += cw_vectors[i]
        return np.max(loads)
    
    def hill_climb(start_mask, max_iters=1000):
        """Perform local search starting from start_mask."""
        current_mask = start_mask
        current_load = calculate_max_load(current_mask)
        
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            for i in range(D):
                # Try flipping bit i
                new_mask = current_mask ^ (1 << i)
                new_load = calculate_max_load(new_mask)
                if new_load < current_load:
                    current_mask = new_mask
                    current_load = new_load
                    improved = True
                    # Break and restart the loop to try other flips from the new state
                    # This is standard "best improvement" or "first improvement" strategy.
                    # First improvement is usually faster per iteration.
                    break 
        return current_mask, current_load
    
    best_mask = 0
    best_load = float('inf')
    
    # Multi-start: Try a few random initializations and one greedy one
    # Greedy initialization: Start with all CW, then flip if it improves
    greedy_mask = 0
    for i in range(D):
        # Try flipping i from CW to CCW
        test_mask = greedy_mask | (1 << i)
        test_load = calculate_max_load(test_mask)
        # Compare with current greedy state load? 
        # Actually, just do a simple hill climb from 0
        pass
    
    # Let's just do 50 random restarts + 1 start from 0
    np.random.seed(42) # For reproducibility in evaluation if needed, though not strictly required
    starts = [0, (1 << D) - 1] # All CW, All CCW
    for _ in range(50):
        starts.append(np.random.randint(0, 1 << D))
        
    for start in starts:
        m, l = hill_climb(start)
        if l < best_load:
            best_load = l
            best_mask = m
            
    # Construct the routing array
    routing = np.zeros(D)
    for i in range(D):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing

# EVOLVE-BLOCK-END
