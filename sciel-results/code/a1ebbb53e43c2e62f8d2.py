import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Vertices 0..19 in A, 20..39 in B
    # Base K(20,20) has 400 edges
    # Add one internal edge (0,1) to get 401 edges
    
    # This gives:
    # - Edge (0,1): common neighbors = all 20 vertices in B. Book = 20.
    # - Edges (0,b) for b in B: common neighbors = {1}. Book = 1.
    # - Edges (1,b) for b in B: common neighbors = {0}. Book = 1.
    # - Other bipartite edges: no common neighbors. Book = 0.
    # Largest book = 20.
    
    edges = []
    
    # Add all bipartite edges
    for u in range(20):
        for v in range(20, 40):
            edges.append([u, v])
    
    # Add one internal edge
    edges.append([0, 1])
    
    # Should be exactly 401 edges
    # 400 + 1 = 401
    
    return edges
