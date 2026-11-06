"""Constructs a dense triangle-free graph on 16 vertices with chromatic number at least 4."""

def construct():
    # Grötzsch graph on vertices 0..10
    # Vertices 0-4 form a cycle C5
    # Vertices 5-9 are copies of 0-4, connected to shifted neighbors
    # Vertex 10 is connected to all 5-9
    
    edges = []
    
    # C5: 0-1, 1-2, 2-3, 3-4, 4-0
    for i in range(5):
        edges.append([i, (i + 1) % 5])
        
    # Mycielski step 1: v'_i connected to v_{i-1}, v_{i+1}
    for i in range(5):
        edges.append([5 + i, (i - 1) % 5])
        edges.append([5 + i, (i + 1) % 5])
        
    # Mycielski step 2: u (10) connected to all v'_i
    for i in range(5):
        edges.append([10, 5 + i])
        
    # Add vertices 11-15
    # Independent set S1 = {5, 6, 7, 8, 9} (size 5)
    S1 = [5, 6, 7, 8, 9]
    # Independent set S2 = {0, 2, 10} (size 3, disjoint from S1)
    # Check: 10 is not connected to 0 or 2. 0 and 2 are not connected in C5.
    S2 = [0, 2, 10]
    
    # Group 1: vertices 11, 12, 13 connected to S1
    group1 = [11, 12, 13]
    for v in group1:
        for u in S1:
            edges.append([v, u])
            
    # Group 2: vertices 14, 15 connected to S2
    group2 = [14, 15]
    for v in group2:
        for u in S2:
            edges.append([v, u])
            
    # Connect Group 1 and Group 2 (complete bipartite between them)
    # This is valid because S1 and S2 are disjoint, so no triangles formed with old vertices
    for v1 in group1:
        for v2 in group2:
            edges.append([v1, v2])
            
    return edges
