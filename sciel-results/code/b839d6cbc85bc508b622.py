import numpy as np
import time

def priority(adjacency):
    n = adjacency.shape[0]
    adj = np.asarray(adjacency, dtype=bool)
    
    # Precompute neighbor lists for faster blocking
    # Use lists of integers for faster indexing? 
    # np.array indexing with a list is similar to array.
    neighbors = [np.where(adj[i])[0] for i in range(n)]
    
    def evaluate(order):
        blocked = np.zeros(n, dtype=bool)
        size = 0
        for v in order:
            if not blocked[v]:
                size += 1
                # Block neighbors
                # Check if neighbors is not empty to avoid overhead of empty assignment?
                # np assignment with empty array is fast anyway.
                blocked[neighbors[v]] = True
                blocked[v] = True
        return size

    start_time = time.time()
    # Safety margin: 10 seconds
    deadline = start_time + 110.0
    
    best_score = -1
    best_order = None
    
    # Strategy: Multiple restarts to explore different basins of attraction.
    # We allocate time to each restart.
    num_restarts = 10
    time_per_restart = (deadline - start_time) / num_restarts
    
    # Initial best guess: Degree based
    degrees = adj.sum(axis=1)
    init_order_deg = np.argsort(degrees)
    best_score = evaluate(init_order_deg)
    best_order = init_order_deg.copy()
    
    for r in range(num_restarts):
        # Choose initial order
        if r == 0:
            # Start from the degree-based order
            current_order = init_order_deg.copy()
        else:
            # Random restarts
            current_order = np.random.permutation(n)
            
        current_score = evaluate(current_order)
        
        if current_score > best_score:
            best_score = current_score
            best_order = current_order.copy()
            
        T = 1.0
        # Cooling rate: We want to stay hot for a while, then cool.
        # With ~10k-50k steps per restart, 0.999 is reasonable.
        cooling = 0.999
        
        restart_deadline = start_time + (r + 1) * time_per_restart
        
        # Local search loop
        while time.time() < restart_deadline:
            i = np.random.randint(n)
            j = np.random.randint(n)
            if i == j:
                continue
            
            # Swap
            current_order[i], current_order[j] = current_order[j], current_order[i]
            
            new_score = evaluate(current_order)
            
            delta = new_score - current_score
            
            # Acceptance probability
            if delta > 0 or np.random.rand() < np.exp(delta / T):
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_order = current_order.copy()
            else:
                # Revert
                current_order[i], current_order[j] = current_order[j], current_order[i]
                
            T *= cooling
            
    # Construct scores
    scores = np.zeros(n)
    for idx, v in enumerate(best_order):
        scores[v] = n - idx
        
    return scores
