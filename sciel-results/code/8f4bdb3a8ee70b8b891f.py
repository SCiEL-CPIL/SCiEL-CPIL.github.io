"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    # Base graph G: 7 vertices, 11 edges, 3-chromatic, triangle-free
    # Vertices 0..4 form C5. Vertices 5, 6 are added.
    # Edges in C5:
    g_edges = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 0],
        # Vertex 5 connected to 0, 2, 3
        [5, 0], [5, 2], [5, 3],
        # Vertex 6 connected to 1, 3, 4
        [6, 1], [6, 3], [6, 4]
    ]
    
    # Mycielski construction
    # Vertices 0..6 are original.
    # Vertices 7..13 are copies (V').
    # Vertex 14 is root (r).
    # Total 15 vertices.
    
    m_edges = []
    # 1. Original edges
    for e in g_edges:
        m_edges.append(e)
        
    # 2. Edges from V to V'
    # For each edge (u, v) in G, add (u, v') and (v, u')
    # v' = v + 7
    for e in g_edges:
        u, v = e
        m_edges.append([u, v + 7])
        m_edges.append([v, u + 7])
        
    # 3. Edges from root r (14) to V'
    for i in range(7):
        m_edges.append([14, i + 7])
        
    # 4. Add 16th vertex (15) connected to all V' (7..13)
    # This adds 7 edges.
    for i in range(7):
        m_edges.append([15, i + 7])
        
    return m_edges
# EVOLVE-BLOCK-END
