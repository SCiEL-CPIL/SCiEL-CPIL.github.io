def construct():
    # The Fano plane is the minimum 3-uniform hypergraph that is 3-chromatic.
    # It has 7 vertices and 7 edges.
    # Vertices 0-6 form the Fano plane. Vertices 7-8 are isolated.
    
    # Fano plane edges
    edges = [
        (0, 1, 2),
        (0, 3, 4),
        (0, 5, 6),
        (1, 3, 5),
        (1, 4, 6),
        (2, 3, 6),
        (2, 4, 5)
    ]
    
    # A valid 3-coloring of the Fano plane.
    # In the Fano plane, any 2-coloring fails.
    # One standard 3-coloring:
    # 0: 0
    # 1: 1
    # 2: 2
    # 3: 0
    # 4: 1
    # 5: 2
    # 6: 0
    # Let's verify:
    # (0,1,2) -> 0,1,2 (OK)
    # (0,3,4) -> 0,0,1 (OK)
    # (0,5,6) -> 0,2,0 (OK)
    # (1,3,5) -> 1,0,2 (OK)
    # (1,4,6) -> 1,1,0 (OK)
    # (2,3,6) -> 2,0,0 (OK)
    # (2,4,5) -> 2,1,2 (OK)
    
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    # Ensure types are strictly Python int and list
    edges = [list(e) for e in edges]
    colors = [int(c) for c in colors]
    
    return {'edges': edges, 'colors': colors}
