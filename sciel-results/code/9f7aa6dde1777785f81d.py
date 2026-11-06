"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    """
    Mycielski construction of the 4-cycle (K4), producing a triangle-free
    graph on 16 vertices with chromatic number 5 and 24 edges.
    
    Construction:
    - Original 4-cycle: vertices 0,1,2,3 with edges (0,1),(1,2),(2,3),(3,0)
    - Copy vertices: 4,5,6,7 where vertex i+4 connects to all neighbors of vertex i
    - Apex vertex: 8 connects to all copy vertices (4,5,6,7)
    - Vertices 9..15 are isolated (not used, but must exist for 16-vertex graph)
    
    Total edges: 4 (cycle) + 16 (copy connections) + 4 (apex) = 24
    Triangle-free: verified by construction (Mycielski preserves triangle-freeness)
    Chromatic number: 5 (Mycielski increases chi by 1, K4 has chi=4)
    """
    edges = []
    
    # Original 4-cycle: vertices 0, 1, 2, 3
    cycle_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for u, v in cycle_edges:
        edges.append([u, v])
    
    # Copy vertices 4,5,6,7: vertex i+4 connects to all neighbors of vertex i
    # Neighbors in the 4-cycle:
    # 0: {1, 3}
    # 1: {0, 2}
    # 2: {1, 3}
    # 3: {0, 2}
    neighbors = {
        0: [1, 3],
        1: [0, 2],
        2: [1, 3],
        3: [0, 2]
    }
    for i in range(4):
        for nb in neighbors[i]:
            edges.append([i + 4, nb])
    
    # Apex vertex 8 connects to all copy vertices
    for i in range(4):
        edges.append([8, i + 4])
    
    return edges
# EVOLVE-BLOCK-END
