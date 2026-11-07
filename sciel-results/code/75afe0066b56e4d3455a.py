def construct():
    # The Fano plane is the unique (up to isomorphism) 3-uniform hypergraph
    # with the minimum number of edges (7) that is 3-chromatic.
    # Any 3-uniform hypergraph with fewer than 7 edges is 2-colorable.
    # We embed the Fano plane in 9 vertices (0..8) by adding 2 isolated vertices.
    
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # Valid 3-coloring:
    # 0: 0, 1: 1, 2: 2
    # 3: 1, 4: 1, 5: 2
    # 6: 0
    # 7: 0, 8: 0 (isolated)
    colors = [0, 1, 2, 1, 1, 2, 0, 0, 0]
    
    return {'edges': edges, 'colors': colors}
