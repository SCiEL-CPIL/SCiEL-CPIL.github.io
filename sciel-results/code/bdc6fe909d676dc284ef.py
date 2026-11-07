"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # Base edges
    base = {tuple(e) for e in P['edges']}
    base.add((e[1], e[0]) for e in P['edges']) # Ensure undirected check
    
    # Add edges from center 12 to specific odd vertices
    # These edges create a hub-and-spoke structure that reduces diameter to <= 4
    # while maintaining triangle-free property (bipartite subgraph added)
    added_edges = []
    center = 12
    targets = [1, 3, 5, 7, 9, 15, 17, 19, 21, 23]
    
    for t in targets:
        u, v = center, t
        if u > v:
            u, v = v, u
        if (u, v) not in base:
            added_edges.append([u, v])
            
    return added_edges
# EVOLVE-BLOCK-END
