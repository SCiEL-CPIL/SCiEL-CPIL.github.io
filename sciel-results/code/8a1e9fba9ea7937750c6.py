import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Preprocess
    n = adjacency.shape[0]
    adj_list = [np.nonzero(adjacency[i])[0] for i in range(n)]
    degrees = np.array([len(neighbors) for neighbors in adj_list])
    
    # Calculate base scores
    # avg_neighbor_degree
    avg_nd = np.zeros(n)
    for i in range(n):
        if degrees[i] > 0:
            avg_nd[i] = np.mean(degrees[adj_list[i]])
        else:
            avg_nd[i] = 0
    
    # Score: Low degree, High neighbor degree
    base_scores = -degrees + 0.5 * avg_nd
    
    # Simulation
    K = 100000 # High K for robustness
    counts = np.zeros(n)
    
    # Pre-generate noise? Or generate on fly?
    # On fly is fine.
    
    for k in range(K):
        if time.time() > deadline:
            break
            
        # Add noise
        noise = np.random.normal(0, 0.1, n)
        scores = base_scores + noise
        
        # Greedy Decoder
        order = np.argsort(-scores)
        blocked = np.zeros(n, dtype=bool)
        
        for v in order:
            if blocked[v]:
                continue
            counts[v] += 1
            blocked[v] = True
            # Block neighbors
            for u in adj_list[v]:
                blocked[u] = True
                
    return counts
