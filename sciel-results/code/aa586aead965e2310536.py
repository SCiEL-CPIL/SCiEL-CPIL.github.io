def construct():
    # Start with C4 (cycle of 4 vertices)
    # Vertices 0,1,2,3
    # Edges (0,1), (1,2), (2,3), (3,0)
    n = 4
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    
    def mycielski(edges, n):
        new_n = 2 * n + 1
        new_edges = []
        # 1. Original edges
        for u, v in edges:
            new_edges.append([u, v])
        # 2. Edges from u to v' if u-v in G
        for u, v in edges:
            new_edges.append([u, v + n])
            new_edges.append([v, u + n])
        # 3. Edges from w to all v'
        w = 2 * n
        for i in range(n):
            new_edges.append([w, i + n])
        return new_edges, new_n
    
    # Apply Mycielski 3 times to get n=39
    # 4 -> 9 -> 19 -> 39
    for _ in range(3):
        edges, n = mycielski(edges, n)
        
    return {'n': n, 'edges': edges}
