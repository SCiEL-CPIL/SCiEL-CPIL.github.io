import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    n = 31
    q = 5
    
    # Generate points of PG(2, 5)
    # Points are 1-dimensional subspaces of F_5^3
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                # Normalize: first non-zero component is 1
                # Or just store tuple and use set of tuples
                # But we need consistent representation
                # Let's use the vector (x,y,z) and divide by first non-zero
                # Actually, just collect all non-zero vectors and group by scalar multiples
                pass
    
    # Better way: iterate through all non-zero vectors and add if not a multiple of existing
    vectors = []
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                vectors.append((x, y, z))
                
    # Group by projective point
    points = []
    used = set()
    for v in vectors:
        if v in used:
            continue
        # Add v and its multiples
        for k in range(1, q):
            mult = ((v[0]*k)%q, (v[1]*k)%q, (v[2]*k)%q)
            used.add(mult)
        points.append(v)
        
    # There should be 31 points
    assert len(points) == 31, f"Expected 31 points, got {len(points)}"
    
    # Define polarity
    # Bilinear form B(u,v) = u0*v0 + u1*v1 + u2*v2 mod 5
    # Edge if B(u,v) == 0 and u != v
    
    edges = set()
    for i in range(n):
        for j in range(i+1, n):
            u = points[i]
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                edges.add((i, j))
                
    # Check number of edges
    # Should be 90
    if len(edges) != 90:
        # Fallback to seed if something goes wrong
        pairs = [[i, j] for i in range(31) for j in range(i + 1, 31)]
        return pairs[:91]
        
    # Count C4s in this graph
    # We need to add 1 edge to get 91
    # We want to add the edge (u,v) that minimizes the number of new C4s
    # New C4s = number of common neighbors of u and v
    
    # Build adjacency list
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    best_edge = None
    min_common = n
    
    for i in range(n):
        for j in range(i+1, n):
            if (i, j) in edges:
                continue
            # Count common neighbors
            common = len(adj[i].intersection(adj[j]))
            if common < min_common:
                min_common = common
                best_edge = (i, j)
                
    if best_edge:
        edges.add(best_edge)
        
    # Convert to list of lists
    result = [[u, v] for u, v in edges]
    
    # Ensure exactly 91 edges
    if len(result) != 91:
         # If we didn't reach 91, add random non-edges
         while len(result) < 91:
             # Find a non-edge
             for i in range(n):
                 for j in range(i+1, n):
                     if (i,j) not in edges:
                         result.append([i,j])
                         edges.add((i,j))
                         break
                 if len(result) >= 91: break
             if len(result) >= 91: break
             
    return result[:91]
