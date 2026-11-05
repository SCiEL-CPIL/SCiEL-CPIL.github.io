import time

def construct():
    start_time = time.time()
    
    # Generate hexagon of radius 3 (37 points)
    R = 3
    points = []
    for a in range(-R, R + 1):
        for b in range(-R, R + 1):
            if abs(a + b) <= R:
                points.append((a, b))
    
    # Identify the 5 points to remove to maximize edges (minimize cost)
    # Based on analysis, removing a full edge of 4 points (including corners) 
    # and one opposite corner yields the optimal 76 edges.
    # Edge points on a=3: (3,0), (3,-1), (3,-2), (3,-3)
    # Opposite corner: (0,3)
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
            result.append(list(p))
            
    return result
