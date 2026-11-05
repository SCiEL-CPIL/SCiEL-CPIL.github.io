import math
import time

def construct():
    deadline = time.time() + 140
    
    def dist_sq(p, q):
        da = p[0] - q[0]
        db = p[1] - q[1]
        return da*da + db*db + da*db

    def count_edges(points):
        count = 0
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if dist_sq(points[i], points[j]) == 1:
                    count += 1
        return count

    # Generate hexagon of radius 3 (37 points)
    # Axial coordinates (a, b) where a+b+c=0, c=-a-b
    # We use a, b as the integer pair.
    # Hexagon of radius R: |a| <= R, |b| <= R, |a+b| <= R
    R = 3
    hex_points = []
    for a in range(-R, R+1):
        for b in range(-R, R+1):
            if abs(a+b) <= R:
                hex_points.append([a, b])
    
    # We have 37 points. Need 32. Remove 5.
    # Greedy removal of boundary points with lowest degree
    points = hex_points
    while len(points) > 32:
        if time.time() > deadline - 5:
            break
        # Calculate degree for each point within the current set
        point_set = set(map(tuple, points))
        min_deg = float('inf')
        min_idx = -1
        for i, p in enumerate(points):
            deg = 0
            # Check 6 neighbors
            for da, db in [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]:
                nb = (p[0]+da, p[1]+db)
                if nb in point_set:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
                min_idx = i
        
        if min_idx != -1:
            points.pop(min_idx)
            
    # Final check and return
    if len(points) != 32:
        # Fallback to a known valid set if greedy failed
        # Use the 16x2 strip as fallback
        points = [[a, b] for a in range(16) for b in range(2)]
        
    return points
