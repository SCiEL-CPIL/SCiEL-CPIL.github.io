def construct():
    """
    Constructs a graph with 32 vertices: 8 disjoint copies of K4.
    - No C5: Each component is K4, which has no C5.
    - No independent set of size 9: Alpha is 8 * 1 = 8.
    """
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    return {'n': n, 'edges': edges}
