def construct():
    # Construct 8 disjoint copies of K4.
    # Each K4 has 4 vertices and 6 edges.
    # Total vertices n = 8 * 4 = 32.
    # Independence number alpha = 8 * 1 = 8.
    # No 5-cycles because max component size is 4.
    
    n = 32
    edges = []
    for block in range(8):
        base = block * 4
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
