def construct():
    # Generate points in a hexagonal spiral
    points = []
    points.append((0, 0))
    
    # Ring 1
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    points.extend(ring1)
    
    # Ring 2
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -1), 
             (-2, 0), (-2, 1), (-2, 2), (-1, 2), (0, 2), (1, 1)]
    points.extend(ring2)
    
    # Ring 3
    ring3 = [(3, 0), (3, -1), (3, -2), (3, -3), (2, -3), (1, -3), 
             (0, -3), (-1, -2), (-2, -1), (-3, 0), (-3, 1), (-3, 2), 
             (-3, 3), (-2, 3), (-1, 3), (0, 3), (1, 2), (2, 1)]
    
    # We have 1 + 6 + 12 = 19 points so far.
    # We need 32 - 19 = 13 more points from ring 3.
    points.extend(ring3[:13])
    
    # Return as list of lists [a, b]
    return [[a, b] for a, b in points]
