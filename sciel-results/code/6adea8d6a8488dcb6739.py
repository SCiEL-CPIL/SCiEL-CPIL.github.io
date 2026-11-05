import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    n = 40
    V1 = list(range(20))
    V2 = list(range(20, 40))
    
    edges = []
    
    # Add bipartite edges for vertices 1..19 in V1 to all vertices in V2
    for u in range(1, 20):
        for v in V2:
            edges.append([u, v])
            
    # Add bipartite edges for vertex 0 in V1 to only two vertices in V2
    # This reduces the common neighbors for edges (0, i)
    edges.append([0, 20])
    edges.append([0, 21])
    
    # Add internal edges: Star centered at 0 with leaves 1..19
    for i in range(1, 20):
        edges.append([0, i])
        
    # Verify edge count
    # Bipartite: 19 * 20 + 2 = 382
    # Internal: 19
    # Total: 401
    
    # Verify distinctness and range
    # All u in 0..19, v in 20..39 or u,v in 0..19
    # No duplicates by construction
    
    return edges
