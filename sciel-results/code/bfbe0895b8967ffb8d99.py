import itertools
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial graph: K_{3,13}
    # U = {0, 1, 2}, V = {3, ..., 15}
    edges = set()
    for u in range(3):
        for v in range(3, 16):
            edges.add((u, v))
            
    # Add all intra-partition edges immediately
    # These are safe because Q3 is bipartite and triangle-free, 
    # and adding edges within partitions creates triangles with common neighbors,
    # or simply doesn't fit the Q3 structure.
    # Specifically, in K_{3,13}, any edge within U or V creates triangles.
    # Q3 is triangle-free.
    for u1 in range(3):
        for u2 in range(u1 + 1, 3):
            edges.add((u1, u2))
    for v1 in range(3, 16):
        for v2 in range(v1 + 1, 16):
            edges.add((v1, v2))

    # Precompute bitmask for each vertex
    adj_mask = [0] * 16
    for (u, v) in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # ... rest of the code for greedy addition of bipartite edges ...
    # Note: We should only add edges between U and V in the greedy loop.
    # Because intra-partition edges are already added.
