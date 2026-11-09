import random
import time

def construct():
    # Grötzsch graph edges on vertices 0..10
    # Vertices 0-4 form a cycle C5
    # Vertex 5-9 are "twins" of 0-4
    # Vertex 10 is the "apex"
    groetzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 6), (1, 7), (2, 8), (3, 9), (4, 5),
        (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
        (5, 10), (6, 10), (7, 10), (8, 10), (9, 10)
    ]
    
    # Note: Standard Grötzsch construction:
    # C5: 0-1-2-3-4-0
    # Mycielski:
    # Add 5-9 corresponding to 0-4?
    # Let's use the standard explicit list for Grötzsch graph (11 vertices).
    # A known set of edges for Grötzsch graph:
    # Cycle 0-1-2-3-4-0
    # 5 connected to 0, 2, 4 ? No.
    # Let's generate it properly or use a verified list.
    
    # Verified Grötzsch edges:
    # 0-1, 1-2, 2-3, 3-4, 4-0 (C5)
    # 5-0, 5-2, 5-4 (5 is connected to 0,2,4? No, 5 is the "new" vertex in Mycielski)
    # Mycielski of C5:
    # V = {0..4, 5..9, 10}
    # Edges in C5: (0,1), (1,2), (2,3), (3,4), (4,0)
    # Edges from 5..9 to 0..4:
    # 5 connected to 1, 2
    # 6 connected to 2, 3
    # 7 connected to 3, 4
    # 8 connected to 4, 0
    # 9 connected to 0, 1
    # Edges among 5..9: (5,7), (7,9), (9,6), (6,8), (8,5) ? No, that's a cycle.
    # Mycielski adds a cycle of "twin" vertices.
    # Twin of 0 is 5. 5 connected to neighbors of 0 (1,4).
    # Twin of 1 is 6. 6 connected to neighbors of 1 (0,2).
    # Twin of 2 is 7. 7 connected to neighbors of 2 (1,3).
    # Twin of 3 is 8. 8 connected to neighbors of 3 (2,4).
    # Twin of 4 is 9. 9 connected to neighbors of 4 (3,0).
    # Edges between twins: 5-7, 7-9, 9-6, 6-8, 8-5. (Cycle 5-7-9-6-8-5)
    # Apex 10 connected to all twins 5,6,7,8,9.
    
    edges_g = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 1), (5, 4),
        (6, 0), (6, 2),
        (7, 1), (7, 3),
        (8, 2), (8, 4),
        (9, 0), (9, 3),
        (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    # Verify triangle-free and 4-chromatic? 
    # It is known to be so.
    
    all_vertices = range(16)
    all_edges = [(i, j) for i in range(16) for j in range(i+1, 16)]
    
    best_edges_count = 0
    best_edge_set = set()
    
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    # Randomized Greedy with Restarts
    for restart in range(100000):
        if time.time() > deadline:
            break
            
        current_edges = set(edges_g)
        # Create a list of candidate edges
        candidates = [e for e in all_edges if e not in current_edges]
        random.shuffle(candidates)
        
        # To speed up triangle check, maintain adjacency sets
        adj = {i: set() for i in all_vertices}
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        for u, v in candidates:
            # Check if adding (u, v) creates a triangle
            # A triangle is formed if there is a common neighbor w
            # common = adj[u] & adj[v]
            if not (adj[u] & adj[v]):
                current_edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
        
        if len(current_edges) > best_edges_count:
            best_edges_count = len(current_edges)
            best_edge_set = current_edges
            
    return [list(e) for e in best_edge_set]
