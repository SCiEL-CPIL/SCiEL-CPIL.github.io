import time

def construct():
    start_time = time.time()
    
    # Generate points of PG(2, 5)
    points = []
    for a in range(5):
        for b in range(5):
            points.append([1, a, b])
    for b in range(5):
        points.append([0, 1, b])
    points.append([0, 0, 1])
    
    n = len(points)
    
    # Build adjacency matrix and edge list
    adj = [[False] * n for _ in range(n)]
    edges = []
    
    for i in range(n):
        p_i = points[i]
        for j in range(i + 1, n):
            p_j = points[j]
            dot = (p_i[0]*p_j[0] + p_i[1]*p_j[1] + p_i[2]*p_j[2]) % 5
            if dot == 0:
                adj[i][j] = True
                adj[j][i] = True
                edges.append([i, j])
                
    # The graph G is C4-free with 90 edges.
    # We need to add one edge to minimize C4s.
    
    # Precompute neighbors
    neighbors = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                neighbors[i].append(j)
                
    best_edge = None
    min_c4s = float('inf')
    
    # Check all non-edges
    for i in range(n):
        for j in range(i + 1, n):
            if not adj[i][j]:
                # Count new C4s formed by adding (i, j)
                # A new C4 is i - j - w - x - i
                # w in N(j), x in N(i)
                # (w, x) must be an edge in G
                # i, j, w, x must be distinct
                
                count = 0
                for w in neighbors[j]:
                    if w == i: continue # w must be distinct from i
                    for x in neighbors[i]:
                        if x == j: continue # x must be distinct from j
                        if w == x: continue # w and x must be distinct
                        if w == i or x == i: continue # w,x distinct from i
                        if w == j or x == j: continue # w,x distinct from j
                        
                        # Check if (w, x) is an edge
                        if adj[w][x]:
                            count += 1
                            
                if count < min_c4s:
                    min_c4s = count
                    best_edge = [i, j]
                    
    if best_edge:
        edges.append(best_edge)
        
    return edges
