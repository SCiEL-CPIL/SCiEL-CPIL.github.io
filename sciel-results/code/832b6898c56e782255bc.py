import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = 110.0  # Safety margin before 150s kill

    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]

    if n == 0:
        return np.array([], dtype=np.float64)

    # Degrees
    degrees = A.sum(axis=1)
    
    # Min-cut inspired scoring:
    # The "cost" of removing vertex v from the graph is the total value of its neighbors.
    # A neighbor u has value proportional to how much independent set potential it represents.
    # Low-degree neighbors are more valuable (they block less), so losing them is costly.
    # High-degree neighbors are less valuable (they block a lot anyway), so losing them is cheaper.
    
    # Value of vertex u: inverse of its degree (low degree = high value)
    # Use 1/(1+deg) to avoid division by zero and keep bounded
    vertex_value = 1.0 / (1.0 + degrees)
    
    # Cost of removing v = sum of values of its neighbors
    # This is A @ vertex_value
    removal_cost = A @ vertex_value
    
    # Score: higher is better. We want low removal cost.
    # score = -removal_cost
    # But also: isolated vertices (degree 0) have removal_cost = 0, score = 0
    # They should be picked first! So we need to boost them.
    
    # Actually, let's think again:
    # If I pick v, I gain v (value = vertex_value[v]) and lose all neighbors (value = removal_cost[v])
    # Net gain = vertex_value[v] - removal_cost[v]
    # But this isn't quite right either, because the neighbors might not have been picked anyway.
    
    # Better: The greedy process means that when we consider picking v,
    # all previously picked vertices' neighbors are already removed.
    # So the "opportunity cost" of picking v is the number of unpicked neighbors
    # that would have been good picks.
    
    # Approximation: score(v) = vertex_value[v] - removal_cost[v]
    # This rewards vertices that are valuable themselves but don't block many valuable neighbors.
    
    scores = vertex_value - removal_cost
    
    # Add a small tie-breaker: prefer lower degree among equal scores
    # (lower degree means less blocking)
    deg_norm = degrees / (degrees.max() + 1e-9)
    scores = scores - 0.01 * deg_norm
    
    # Safety: ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback to simple degree-based score
        return -degrees.astype(np.float64)
    
    return scores.astype(np.float64)
