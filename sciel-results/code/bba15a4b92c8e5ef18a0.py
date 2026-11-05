import time

def construct():
    # Theoretical maximum for n=32 is 76 edges (Harborth's bound).
    # A hexagon of radius 3 has 37 points and 90 edges.
    # Removing 5 specific points reduces the edge count to 76.
    # The points removed are (3,0), (3,-1), (3,-2), (3,-3) and (0,3).
    
    R = 3
    points = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if abs(a + b) <= R:
                points.append((a, b))
    
    remove_set = {
        (3, 0),
        (3, -1),
        (3, -2),
        (3, -3),
        (0, 3)
    }
    
    result = []
    for p in points:
        if p not in remove_set:
            result.append([p[0], p[1]])
            
    return result
