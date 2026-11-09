def construct():
    # Fano plane on vertices 0-6, embedded in 9 vertices (7,8 isolated)
    edges = [
        [0, 1, 2],
        [0, 3, 4],
        [0, 5, 6],
        [1, 3, 5],
        [1, 4, 6],
        [2, 3, 6],
        [2, 4, 5]
    ]
    
    # 3-coloring: 
    # Vertices 0,3,6 -> color 0
    # Vertices 1,4 -> color 1
    # Vertices 2,5 -> color 2
    # Vertices 7,8 -> color 0 (isolated, can be any color)
    colors = [0, 1, 2, 0, 1, 2, 0, 0, 0]
    
    return {'edges': edges, 'colors': colors}
