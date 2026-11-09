"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    # Vertices 0..23
    # Bipartite sets:
    # Even: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22
    # Odd:  1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23
    
    # Strategy: Connect all vertices to a central hub pair (11, 12) to ensure low diameter.
    # 11 is Odd, 12 is Even.
    # Edges must be between Even and Odd.
    # Connect all Even vertices (except 12) to 11.
    # Connect all Odd vertices (except 11) to 12.
    
    evens = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    odds = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    
    added_edges = []
    
    # Connect Evens to 11 (Odd hub)
    for u in evens:
        if u == 12:
            continue
        added_edges.append([u, 11])
        
    # Connect Odds to 12 (Even hub)
    for v in odds:
        if v == 11:
            continue
        added_edges.append([12, v])
        
    # This results in 11 + 11 = 22 edges.
    # Diameter is 3 (max path u-11-12-v).
    # Triangle-free (bipartite).
    # Connected.
    
    return added_edges
# EVOLVE-BLOCK-END
