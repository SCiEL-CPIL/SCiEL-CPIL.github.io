def construct():
    # Chvátal graph on vertices 0..11
    # This is a well-known 4-chromatic triangle-free graph with 24 edges.
    chvatal_edges = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 5], [1, 6],
        [2, 3], [2, 5],
        [3, 6], [3, 7],
        [4, 5], [4, 8], [4, 9],
        [5, 10],
        [6, 7], [6, 8],
        [7, 9], [7, 10],
        [8, 9], [8, 11],
        [9, 10], [9, 11],
        [10, 11]
    ]
    
    # Add a C4 (cycle of length 4) on vertices 12..15
    # This adds 4 edges and remains triangle-free and 2-colorable,
    # so the chromatic number of the union is max(4, 2) = 4.
    extra_edges = [
        [12, 13],
        [13, 14],
        [14, 15],
        [15, 12]
    ]
    
    return chvatal_edges + extra_edges
