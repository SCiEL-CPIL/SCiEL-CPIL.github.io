# EVOLVE-BLOCK-START
import numpy as np

def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem using Local Search with Random Restarts.
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
    
    # Precompute edge index lists for CW and CCW for each demand
    # cw_edges[i] is a list of edge indices
    # ccw_edges[i] is a list of edge indices
    
    cw_edges = []
    ccw_edges = []
    
    for s, t, w in demands:
        cw_list = []
        ccw_list = []
        
        # Clockwise from s to t: edges s, s+1, ..., t-1 (mod n)
        # If s <= t: s to t-1
        # If s > t: s to n-1 and 0 to t-1
        
        if s <= t:
            for e in range(s, t):
                cw_list.append(e)
        else:
            for e in range(s, n):
                cw_list.append(e)
            for e in range(0, t):
                cw_list.append(e)
        
        # Counter-clockwise is the complement
        cw_set = set(cw_list)
        for e in range(n):
            if e not in cw_set:
                ccw_list.append(e)
                
        cw_edges.append(cw_list)
        ccw_edges.append(ccw_list)
        
    # Weights array
    weights = np.array([w for s, t, w in demands])
    
    def calc_max_load(routing_mask):
        """
        Calculate max load for a given routing mask (integer or array of 0/1).
        routing_mask[i] = 0 -> CW, 1 -> CCW
        """
        loads = np.zeros(n)
        for i in range(D):
            if (routing_mask >> i) & 1:
                # CCW
                for e in ccw_edges[i]:
                    loads[e] += weights[i]
            else:
                # CW
                for e in cw_edges[i]:
                    loads[e] += weights[i]
        return np.max(loads)
    
    best_max_load = float('inf')
    best_mask = 0
    
    # Local Search with Random Restarts
    # Number of restarts
    num_restarts = 1000
    
    for _ in range(num_restarts):
        # Random initialization
        current_mask = np.random.randint(0, 1 << D)
        current_load = calc_max_load(current_mask)
        
        if current_load < best_max_load:
            best_max_load = current_load
            best_mask = current_mask
            
        # Hill Climbing
        improved = True
        while improved:
            improved = False
            for i in range(D):
                # Try flipping bit i
                new_mask = current_mask ^ (1 << i)
                new_load = calc_max_load(new_mask)
                if new_load < current_load:
                    current_mask = new_mask
                    current_load = new_load
                    improved = True
                    if current_load < best_max_load:
                        best_max_load = current_load
                        best_mask = current_mask
                        # Early break if we hit theoretical lower bound?
                        # Lower bound is max(weights) ? No.
                        # Lower bound is ceil(total_weight / n) = 4.
                        # But specific cuts impose higher bounds.
                        # If we reach 7 or 8, it's likely optimal.
                        pass
        
        # Optional: Simulated Annealing step? 
        # For now, pure hill climbing with restarts is a different family.
        # To make it more robust, we could add a "shake" or SA.
        # But 1000 restarts of hill climbing on 15 bits should be very good.
        
    # Construct the routing array
    routing = np.zeros(D)
    for i in range(D):
        if (best_mask >> i) & 1:
            routing[i] = 1.0
        else:
            routing[i] = 0.0
            
    return routing

# EVOLVE-BLOCK-END
