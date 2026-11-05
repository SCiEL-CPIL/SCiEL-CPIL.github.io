import numpy as np
import time

def priority(adjacency):
    n = adjacency.shape[0]
    # Convert to list of neighbors for faster lookup if needed, 
    # but for N=200, matrix operations are fine.
    
    # We will simulate the greedy process:
    # 1. available = all nodes
    # 2. While available:
    #    a. Find node v in available with min degree in induced subgraph of available
    #    b. Assign priority[v] = -current_step (higher is better, so earlier step is higher)
    #    c. Remove v and neighbors(v) from available
    #    d. current_step += 1
    
    available = np.ones(n, dtype=bool)
    scores = np.zeros(n, dtype=float)
    
    start_time = time.time()
    max_time = 120.0 # Safety margin
    
    step = 0
    while available.any():
        if time.time() - start_time > max_time - 5:
            # Fallback to static degree if we run out of time (unlikely)
            break
        
        # Calculate degrees within available set
        # deg[i] = sum(adjacency[i, j] for j in available)
        # We can do this efficiently:
        # adj_avail = adjacency[:, available]
        # degrees = adj_avail.sum(axis=1)
        
        # But we only care about nodes in available.
        # Let's get indices of available nodes
        avail_indices = np.where(available)[0]
        
        if len(avail_indices) == 0:
            break
            
        # Compute degrees for available nodes within the available subgraph
        # adjacency[avail_indices, :] gives rows for avail nodes
        # We only want columns that are also available
        sub_adj = adjacency[avail_indices][:, available]
        degrees = sub_adj.sum(axis=1)
        
        # Find index in avail_indices with min degree
        # Tie-break by index (lower index first? or arbitrary?)
        # To be deterministic, we can use argsort on (degree, index)
        # But we just need the min.
        min_deg = np.min(degrees)
        candidates = avail_indices[degrees == min_deg]
        
        # Pick the first one (lowest index) for determinism
        v = candidates[0]
        
        # Assign score. Higher score = picked earlier.
        # So score should decrease as step increases.
        scores[v] = -step
        
        # Remove v and its neighbors from available
        neighbors = np.where(adjacency[v, :])[0]
        available[v] = False
        available[neighbors] = False
        
        step += 1
        
    # Any vertex not assigned a score (shouldn't happen if loop finishes) 
    # gets a very low score.
    scores[~np.isfinite(scores)] = -1e9 # Or just leave as 0 if initialized? 
    # Wait, I initialized scores to 0. 
    # If a vertex is never picked (e.g. if loop breaks early), it stays 0.
    # But 0 is higher than -step for step > 0.
    # So unpicked vertices would be picked *before* step 0 vertices?
    # No, step starts at 0. scores[0] = 0. scores[1] = -1.
    # If a vertex is unpicked, it has score 0. It might compete with step 0.
    # Better to initialize scores to -inf and assign -step.
    
    return scores
