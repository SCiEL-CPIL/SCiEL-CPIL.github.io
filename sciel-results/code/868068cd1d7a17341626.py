def construct():
    """
    Constructs a graph on 32 vertices with no 5-cycle and no independent set of size 9.
    The graph is a disjoint union of 8 complete graphs K4.
    
    - Each K4 has 4 vertices and 6 edges.
    - Total vertices n = 8 * 4 = 32.
    - Total edges = 8 * 6 = 48.
    - A K4 does not contain a 5-cycle. Since the components are disconnected, no 5-cycle can span multiple components.
    - The independence number of a K4 is 1.
    - The independence number of the disjoint union is the sum of the independence numbers of the components: 8 * 1 = 8.
    - 8 < 9, so the condition is satisfied.
    """
    n = 32
    edges = []
    for b in range(8):
        base = 4 * b
        # Add all edges for K4 on vertices base, base+1, base+2, base+3
        for i in range(4):
            for j in range(i + 1, 4):
                edges.append([base + i, base + j])
    
    return {'n': n, 'edges': edges}
