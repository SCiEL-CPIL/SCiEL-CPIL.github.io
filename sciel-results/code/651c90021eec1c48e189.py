import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    # Safety margin: aim to finish well before 150s
    # For N=200, this will take milliseconds, so this is just a formality
    # but required by the directive.
    
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=bool)
        
    n = adjacency.shape[0]
    
    # 1. Compute degrees
    degrees = adjacency.sum(axis=1)
    
    # 2. Compute weights based on Caro-Wei bound approximation
    # w_i = 1 / (1 + deg_i)
    weights = 1.0 / (1.0 + degrees)
    
    # 3. Compute sum of weights of neighbors
    # A is boolean, weights are float. 
    # neighbor_sum[v] = sum_{u in N(v)} weights[u]
    neighbor_sum = adjacency.astype(float) @ weights
    
    # 4. Compute total "cost" of the closed neighborhood
    # Cost(v) = weight(v) + sum_{u in N(v)} weight(u)
    cost = weights + neighbor_sum
    
    # 5. Priority is negative cost (we want to minimize cost)
    priority_scores = -cost
    
    # 6. Add tiny tie-breaker to ensure deterministic ordering
    # Lower index gets slightly higher priority if scores are equal
    # We subtract a small value proportional to index
    tie_breaker = np.arange(n) * 1e-9
    priority_scores = priority_scores - tie_breaker
    
    # Check time (though it should be negligible)
    if time.time() - start_time > 140:
        # Fallback to simple degree if something went wrong (unlikely)
        return -degrees.astype(float)
        
    return priority_scores
