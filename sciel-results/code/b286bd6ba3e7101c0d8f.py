def get_groetzsch_edges():
    edges = []
    # Cycle 0-4
    for i in range(5):
        edges.append((i, (i+1)%5))
    # Mycielski vertices 5-9
    # 5 corresponds to 0. Neighbors of 0 are 1, 4.
    # 6 corresponds to 1. Neighbors of 1 are 0, 2.
    # 7 corresponds to 2. Neighbors of 2 are 1, 3.
    # 8 corresponds to 3. Neighbors of 3 are 2, 4.
    # 9 corresponds to 4. Neighbors of 4 are 3, 0.
    
    mapping = {
        0: [1, 4],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3, 0]
    }
    
    for i in range(5):
        v_new = 5 + i
        for neighbor in mapping[i]:
            edges.append((v_new, neighbor))
            
    # Vertex 10 connected to 5-9
    for i in range(5):
        edges.append((10, 5+i))
        
    # Normalize
    norm_edges = set()
    for u, v in edges:
        if u > v: u, v = v, u
        norm_edges.add((u, v))
    return list(norm_edges)
