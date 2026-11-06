# EVOLVE-BLOCK-START
import numpy as np
import heapq


def priority(adjacency):
    """
    Assign a score to each vertex such that a greedy decoder picking vertices
    in descending score order (skipping neighbors of already-picked vertices)
    yields a large independent set.
    
    Strategy:
    Use K-core decomposition (peeling) to identify peripheral vertices.
    Vertices with low core numbers (removed early in peeling) are in the periphery
    and are generally better candidates for an independent set as they block fewer
    high-value (central) vertices.
    
    We compute the order in which vertices are removed by the k-core peeling process.
    Earlier removal (lower rank) -> Higher priority (higher score).
    We use a min-heap to simulate the peeling process efficiently.
    
    Score = - (peeling_rank + 0.1 * degree)
    """
    adj = np.asarray(adjacency, dtype=np.int32)
    n = adj.shape[0]
    if n == 0:
        return np.array([])
    
    # Compute degrees
    degrees = adj.sum(axis=1)
    
    # Min-heap: (degree, node_id)
    heap = []
    for i in range(n):
        heapq.heappush(heap, (degrees[i], i))
        
    current_degrees = degrees.copy()
    in_graph = np.ones(n, dtype=bool)
    removal_order = np.zeros(n, dtype=np.int32)
    time = 0
    
    while heap:
        deg, u = heapq.heappop(heap)
        
        # Check if this entry is stale
        if not in_graph[u] or current_degrees[u] != deg:
            continue
            
        # Remove u
        in_graph[u] = False
        removal_order[u] = time
        time += 1
        
        # Update neighbors
        # Using np.nonzero on a row of the adjacency matrix
        neighbors = np.nonzero(adj[u])[0]
        for v in neighbors:
            if in_graph[v]:
                current_degrees[v] -= 1
                heapq.heappush(heap, (current_degrees[v], v))
                
    # Calculate scores
    # Higher priority for lower removal_order (earlier removal)
    # Add a small degree penalty to break ties and refine
    scores = - (removal_order.astype(np.float64) + 0.1 * degrees.astype(np.float64))
    
    return scores
# EVOLVE-BLOCK-END
