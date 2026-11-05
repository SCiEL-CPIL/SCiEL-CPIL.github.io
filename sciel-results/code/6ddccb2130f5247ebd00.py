import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0
    
    # We construct a specific 16-vertex triangle-free graph with 45 edges.
    # This is an extension of the Grötzsch graph (11 vertices, 20 edges).
    
    # Vertices:
    # 0-4: Cycle C5 (V)
    # 5-9: Independent set W (copies of V)
    # 10: Apex Z
    # 11-15: New independent set X
    
    edges = []
    
    # 1. Grötzsch Graph Edges
    
    # Cycle on V (0-4)
    for i in range(5):
        edges.append([i, (i + 1) % 5])
        
    # Cross edges between V and W
    # v_i connects to w_{i-1} and w_{i+1}
    # w_j is at index 5 + j
    for i in range(5):
        w_idx_1 = 5 + (i - 1) % 5
        w_idx_2 = 5 + (i + 1) % 5
        edges.append([i, w_idx_1])
        edges.append([i, w_idx_2])
        
    # Apex Z (10) connects to all W (5-9)
    for i in range(5):
        edges.append([10, 5 + i])
        
    # 2. Extension Edges
    # New vertices X (11-15) connect to all W (5-9)
    # This adds 5 * 5 = 25 edges.
    # This maintains triangle-free property because W is an independent set.
    for x in range(11, 16):
        for w in range(5, 10):
            edges.append([x, w])
            
    # Total edges: 5 (cycle) + 10 (cross) + 5 (apex) + 25 (new) = 45
    
    # Verify within time limit (trivial for static construction)
    if time.time() > deadline:
        # Fallback to Clebsch if something goes wrong (unlikely)
        return []
        
    return edges
