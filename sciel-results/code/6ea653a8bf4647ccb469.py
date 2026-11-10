import time
import random

def construct():
    # Grötzsch graph edges (0..10)
    # Cycle 0-1-2-3-4-0
    # 5 connected to 1,3
    # 6 connected to 2,4
    # 7 connected to 3,0
    # 8 connected to 4,1
    # 9 connected to 0,2
    # 10 connected to 5,6,7,8,9
    # Wait, let's use the standard Mycielski construction or a known list.
    # Standard Grötzsch graph edges:
    # 0-1, 1-2, 2-3, 3-4, 4-0
    # 5-1, 5-3
    # 6-2, 6-4
    # 7-3, 7-0
    # 8-4, 8-1
    # 9-0, 9-2
    # 10-5, 10-6, 10-7, 10-8, 10-9
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (5, 1), (5, 3),
        (6, 2), (6, 4),
        (7, 3), (7, 0),
        (8, 4), (8, 1),
        (9, 0), (9, 2),
        (10, 5), (10, 6), (10, 7), (10, 8), (10, 9)
    ]
    
    n = 16
    # Initialize masks
    # mask[i] is a bitmask of neighbors of i
    
    best_edge_count = -1
    best_edges = []
    
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    while time.time() < deadline:
        masks = [0] * n
        current_edges = []
        
        # Add Grötzsch edges
        for u, v in edges:
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            current_edges.append((u, v))
            
        # Generate candidate edges
        candidates = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)): # Not already an edge
                    # Check if adding creates triangle in INITIAL graph
                    # Actually, we just need to check if they share a neighbor in the INITIAL graph?
                    # No, we want to add as many as possible.
                    # If they don't share a neighbor in the initial graph, they are "potentially safe".
                    # But we must re-check against the growing graph.
                    # To reduce candidates, we can just include all non-edges.
                    # There are 120 * 119 / 2 = 7140 total edges.
                    # Grötzsch has ~20.
                    # So ~7100 candidates.
                    candidates.append((u, v))
                    
        random.shuffle(candidates)
        
        for u, v in candidates:
            # Check if adding (u,v) creates a triangle in CURRENT graph
            # Triangle exists if common neighbor count > 0
            if masks[u] & masks[v]:
                continue # Would create triangle
            
            # Add edge
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            current_edges.append((u, v))
            
        if len(current_edges) > best_edge_count:
            best_edge_count = len(current_edges)
            best_edges = current_edges[:]
            
    return [list(e) for e in best_edges]
