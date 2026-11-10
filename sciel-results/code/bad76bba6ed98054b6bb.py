import random
import time

def construct():
    # Grötzsch graph edges on vertices 0..10
    # This is a triangle-free graph with chromatic number 4.
    groetzsch_edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
        (0, 6), (1, 7), (2, 8), (3, 9), (4, 5),
        (1, 6), (2, 7), (3, 8), (4, 9), (0, 5),
        (5, 10), (6, 10), (7, 10), (8, 10), (9, 10)
    ]
    
    n = 16
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    groetzsch_set = set(groetzsch_edges)
    
    best_edges = [list(e) for e in groetzsch_edges]
    best_count = len(groetzsch_edges)
    
    start_time = time.time()
    # Allow 110 seconds to be safe
    deadline = start_time + 110.0
    
    while time.time() < deadline:
        # Initialize with the core
        current_edges = set(groetzsch_set)
        adj = [set() for _ in range(n)]
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Candidates are edges not in the core
        # We can filter this once outside the loop if we want, but it's small
        candidates = [e for e in all_edges if e not in current_edges]
        random.shuffle(candidates)
        
        for u, v in candidates:
            # Check if u and v have a common neighbor
            # If they do, adding (u,v) creates a triangle
            if adj[u].isdisjoint(adj[v]):
                current_edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                
        if len(current_edges) > best_count:
            best_count = len(current_edges)
            best_edges = [list(e) for e in current_edges]
            
    return best_edges
