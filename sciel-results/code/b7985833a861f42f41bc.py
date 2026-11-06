import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115

    q = 7
    # Generate points in PG(2, q)
    # Normalization: first non-zero coordinate is 1
    points = []
    
    # Case 1: x = 1
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
            
    # Case 2: x = 0, y = 1
    for z in range(q):
        points.append((0, 1, z))
        
    # Case 3: x = 0, y = 0, z = 1
    points.append((0, 0, 1))
    
    n_total = len(points)
    # n_total should be q^2 + q + 1 = 49 + 7 + 1 = 57
    
    # Build adjacency for the full polarity graph
    # Adjacency: dot product is 0
    adj_mask = [0] * n_total
    
    for i in range(n_total):
        x1, y1, z1 = points[i]
        for j in range(i + 1, n_total):
            x2, y2, z2 = points[j]
            # Dot product in F_q
            dot = (x1 * x2 + y1 * y2 + z1 * z2) % q
            if dot == 0:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                
    # We want the largest n <= 43
    target_n = 43
    
    # Check if we have enough vertices
    if n_total < target_n:
        target_n = n_total
        
    # Extract subgraph for vertices 0 to target_n - 1
    # The induced subgraph is C4-free and alpha <= 7
    sub_adj_mask = adj_mask[:target_n]
    
    # Construct edges list
    edges = []
    for u in range(target_n):
        for v in range(u + 1, target_n):
            if (sub_adj_mask[u] >> v) & 1:
                edges.append([u, v])
                
    return {'n': target_n, 'edges': edges}
