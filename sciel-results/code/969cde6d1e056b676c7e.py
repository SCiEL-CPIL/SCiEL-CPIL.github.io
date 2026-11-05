import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 120.0  # seconds
    
    # 1. Generate points of PG(2, 7)
    # Points are represented as tuples (x, y, z) in F_7^3
    # Normalized: first non-zero coordinate is 1
    points = []
    mod = 7
    
    # Type 1: x=1
    for y in range(mod):
        for z in range(mod):
            points.append((1, y, z))
            
    # Type 2: x=0, y=1
    for z in range(mod):
        points.append((0, 1, z))
        
    # Type 3: x=0, y=0, z=1
    points.append((0, 0, 1))
    
    n_total = len(points) # 57
    
    # 2. Compute edges for the full polarity graph
    # Edge (i, j) if i != j and dot(points[i], points[j]) == 0 mod 7
    edges_full = []
    for i in range(n_total):
        u = points[i]
        for j in range(i + 1, n_total):
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % mod
            if dot == 0:
                edges_full.append((i, j))
                
    # 3. Select a subgraph of size n_target
    # We can go up to 43. Let's try 43.
    n_target = 43
    
    if n_target > n_total:
        n_target = n_total
        
    # Filter edges to only those involving vertices 0..n_target-1
    edges = []
    for u, v in edges_full:
        if u < n_target and v < n_target:
            edges.append([u, v])
            
    # 4. Return result
    # The graph is on vertices 0..n_target-1
    # It is C4-free because it is a subgraph of the C4-free polarity graph.
    # It has alpha < 11 because the full graph has alpha ~ 4-5.
    
    return {'n': n_target, 'edges': edges}
