def construct():
    # Vertices 0, 1, 2 in A; 3..15 in B
    edges = []
    
    # Add all edges between A and B (K_{3,13})
    for a in range(3):
        for b in range(3, 16):
            edges.append([a, b])
            
    # Add a cycle C_{13} on B to maximize edges while keeping max degree <= 2
    # This ensures G[B] is K_{1,3}-free, thus G is Q3-free
    b_vertices = list(range(3, 16))
    for i in range(len(b_vertices)):
        u = b_vertices[i]
        v = b_vertices[(i + 1) % len(b_vertices)]
        edges.append([u, v])
        
    return edges
