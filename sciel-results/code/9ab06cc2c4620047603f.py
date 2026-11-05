import numpy as np
import time

def priority(adjacency):
    """
    Computes a priority score for each vertex for a greedy Maximum Independent Set decoder.
    Strategy: Use Core Number as the primary heuristic. Vertices with lower core numbers
    are in the periphery of the graph and are less constrained, making them better candidates
    for early selection in a greedy MIS algorithm.
    """
    start_time = time.time()
    deadline = 120.0  # Safety margin, though computation is fast
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    # 1. Compute initial degrees
    degrees = adjacency.sum(axis=1).astype(int)
    
    # 2. Compute Core Numbers using the peeling algorithm
    # Since N is small (200), an O(N^2) loop is acceptable and simple.
    # For larger N, we would use a bucket queue, but this is robust for the task constraints.
    remaining = np.ones(n, dtype=bool)
    current_deg = degrees.copy()
    core_numbers = np.zeros(n, dtype=int)
    
    # Precompute neighbor lists for faster update? 
    # adjacency is a bool matrix. np.where is efficient enough for N=200.
    
    for _ in range(n):
        # Check deadline (though unlikely to be hit)
        if time.time() > deadline - 5.0:
            # Fallback to simple degree if we are running out of time
            break
            
        # Find vertex with minimum current degree among remaining
        # Use a mask to ignore removed vertices
        masked_deg = np.where(remaining, current_deg, np.inf)
        v = np.argmin(masked_deg)
        
        if not remaining[v]:
            break
            
        core_numbers[v] = current_deg[v]
        remaining[v] = False
        
        # Update degrees of neighbors
        # Get indices of neighbors that are still remaining
        neighbors = np.where(adjacency[v] & remaining)[0]
        if len(neighbors) > 0:
            current_deg[neighbors] -= 1
            
    # 3. Compute sum of neighbor degrees for tie-breaking
    # sum_neigh_deg[v] = sum(degrees[u] for u in N(v))
    # This can be computed as adjacency @ degrees
    neigh_deg_sums = (adjacency @ degrees).astype(float)
    
    # 4. Construct Score
    # We want lower scores to be picked first? 
    # The problem says: "vertices are selected greedily by descending score".
    # So higher score = picked first.
    # We want to pick Low Core Number, Low Degree, Low NeighDegSum first.
    # So we should negate these values.
    
    # Weighting:
    # Core number is the most important. Range 0..N.
    # Degree is next. Range 0..N.
    # NeighDegSum is small tie-breaker.
    
    # If we use - (core * 1000 + deg), then low core/high negation -> high score.
    # Example:
    # v1: core=0, deg=5 -> score = -(0 + 5) = -5
    # v2: core=1, deg=2 -> score = -(1000 + 2) = -1002
    # Descending order: -5 > -1002. So v1 is picked first. Correct.
    
    scores = -(core_numbers.astype(float) * 1000.0 + degrees.astype(float) + 0.01 * neigh_deg_sums)
    
    return scores

