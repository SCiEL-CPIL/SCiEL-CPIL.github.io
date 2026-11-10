def construct():
    # Directions in the triangular lattice basis (a, b)
    # These correspond to the 6 nearest neighbors of the origin
    dirs = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
    
    points = [(0, 0)]
    a, b = 0, 0
    i = 0
    
    while len(points) < 32:
        # Calculate the number of steps for this segment of the spiral
        side_len = (i // 2) + 1
        dir_idx = i % 6
        da, db = dirs[dir_idx]
        
        for _ in range(side_len):
            if len(points) >= 32:
                break
            a += da
            b += db
            points.append((a, b))
            
        i += 1
        
    return [list(p) for p in points]
