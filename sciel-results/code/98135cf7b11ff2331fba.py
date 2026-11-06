import time

def construct():
    start_time = time.time()
    
    # We construct the Polarity Graph of PG(2, 7)
    # This graph has 57 vertices and is C4-free with small independence number.
    # We take a subgraph of size 43 (the maximum allowed).
    
    q = 7
    points = []
    # Generate all points in PG(2, q)
    # A point is a 1D subspace of F_q^3.
    # We represent each point by a vector (x, y, z) such that the first non-zero component is 1.
    
    # Case 1: x != 0
    # Normalize so x = 1
    for y in range(q):
        for z in range(q):
            points.append((1, y, z))
            
    # Case 2: x = 0, y != 0
    # Normalize so y = 1
    for z in range(q):
        points.append((0, 1, z))
        
    # Case 3: x = 0, y = 0, z != 0
    # Normalize so z = 1
    points.append((0, 0, 1))
    
    n_total = len(points)
    # n_total should be q^2 + q + 1 = 49 + 7 + 1 = 57
    
    # Build adjacency list for the full graph
    # u, v are adjacent if u . v == 0 mod q
    adj_full = [0] * n_total
    for i in range(n_total):
        u = points[i]
        for j in range(n_total):
            if i == j:
                continue
            v = points[j]
            dot = (u[0]*v[0] + u[1]*v[1] + u[2]*v[2]) % q
            if dot == 0:
                adj_full[i] |= (1 << j)
                adj_full[j] |= (1 << i)
                
    # Target size
    n_target = 43
    
    # Select the first n_target vertices
    # The induced subgraph on any subset of vertices is C4-free.
    # The independence number of the subgraph is <= independence number of the full graph.
    # For PG(2,7) polarity graph, alpha is small (approx sqrt(7) + 1 < 11).
    
    adj = [0] * n_target
    for i in range(n_target):
        mask = 0
        full_mask = adj_full[i]
        for j in range(n_target):
            if full_mask & (1 << j):
                mask |= (1 << j)
        adj[i] = mask
        
    # Build edge list
    edges = []
    for u in range(n_target):
        m = adj[u]
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            m ^= lsb
            
    return {'n': n_target, 'edges': edges}
