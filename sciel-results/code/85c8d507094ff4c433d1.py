def construct():
    points = []
    for x in range(7):
        for y in range(5):
            points.append((x, y))
    
    # Remove 4 corners
    points.remove((0, 0))
    points.remove((6, 0))
    points.remove((0, 4))
    points.remove((6, 4))
    
    # Remove 1 edge point (middle of bottom edge)
    points.remove((3, 0))
    
    return points
