def construct():
    n = 16
    edges = []
    # Center is 0
    # 7 Triangles: (0, 1, 2), (0, 3, 4), ..., (0, 13, 14)
    # 1 Leaf: 15 connected to 0
    
    # Triangles
    v = 1
    for _ in range(7):
        a = v
        b = v + 1
        # Edges: (0, a), (a, b), (b, 0)
        edges.append([0, a])
        edges.append([a, b])
        edges.append([b, 0])
        v += 2
        
    # Leaf
    edges.append([0, 15])
    
    return edges
