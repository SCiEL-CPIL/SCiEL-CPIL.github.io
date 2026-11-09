import numpy as np
import heapq
import time

def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    # Convert to list of lists of neighbors for faster iteration if needed, 
    # but for N=200, boolean matrix indexing is fine.
    # However, iterating over neighbors is needed for the peeling step.
    
    degrees = adjacency.sum(axis=1).astype(int)
    
    # Min-heap of (degree, vertex)
    heap = []
    for i in range(n):
        heapq.heappush(heap, (degrees[i], i))
        
    alive = [True] * n
    core = [0] * n
    
    # To handle the "stale" entries in the heap, we just check if the degree matches
    # and if the vertex is alive.
    
    while heap:
        d, u = heapq.heappop(heap)
        
        # Check if this entry is stale
        if not alive[u] or d != degrees[u]:
            continue
            
        alive[u] = False
        core[u] = d
        
        # Update neighbors
        # Get neighbors of u
        # adjacency is a bool matrix. 
        # neighbors = np.where(adjacency[u])[0]
        # This is efficient enough for N=200.
        
        neighbors = np.where(adjacency[u])[0]
        for v in neighbors:
            if alive[v]:
                degrees[v] -= 1
                heapq.heappush(heap, (degrees[v], v))
                
    # Create scores
    # We want high score for low core number.
    # Tie break with low degree (original degree or final? Original is fine, or just core).
    # Let's use original degree for tie-breaking within same core.
    orig_degrees = adjacency.sum(axis=1).astype(int)
    
    # Score = - (core * 1000 + orig_degree)
    # This ensures vertices with lower core number are picked first.
    # Among same core number, lower degree is picked first.
    
    scores = -(np.array(core) * 1000 + orig_degrees).astype(float)
    
    return scores
