def construct():
    n = 15
    edges = []
    
    # A = 0..6, B = 7..14
    # Intra-A: K_7
    for i in range(7):
        for j in range(i+1, 7):
            edges.append([i, j])
    
    # Cross: K_{7,8}
    for i in range(7):
        for j in range(7, 15):
            edges.append([i, j])
    
    # Intra-B: complement of perfect matching on {7..14}
    # Perfect matching: (7,8), (9,10), (11,12), (13,14)
    matching = {(7,8), (9,10), (11,12), (13,14)}
    for i in range(7, 15):
        for j in range(i+1, 15):
            if (i, j) not in matching:
                edges.append([i, j])
    
    return edges
