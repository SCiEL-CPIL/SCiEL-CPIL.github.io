import itertools
import time

def calculate_score(points):
    # points is a list of (x, y)
    # Calculate multiplicity of each non-zero doubled area
    areas = {}
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                det = abs((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
                if det > 0:
                    areas[det] = areas.get(det, 0) + 1
    if not areas:
        return 0
    return max(areas.values())

def construct():
    start_time = time.time()
    deadline = start_time + 45.0
    
    # Candidate 1: 6x5 Grid
    grid_6x5 = [(x, y) for x in range(6) for y in range(5)]
    
    # Candidate 2: 7x5 Grid minus 5 points
    # Full 7x5
    full_7x5 = [(x, y) for x in range(7) for y in range(5)]
    # Remove corners and one edge point
    to_remove = [(0, 0), (6, 0), (0, 4), (6, 4), (0, 2)]
    grid_7x5_mod = [p for p in full_7x5 if p not in to_remove]
    
    # Candidate 3: 10x3 Grid
    grid_10x3 = [(x, y) for x in range(10) for y in range(3)]
    
    # Candidate 4: 5x6 Grid (same as 6x5)
    
    # Candidate 5: 8x4 Grid minus 2 points
    full_8x4 = [(x, y) for x in range(8) for y in range(4)]
    to_remove_8x4 = [(0, 0), (7, 3)] # Remove 2 corners
    grid_8x4_mod = [p for p in full_8x4 if p not in to_remove_8x4]
    
    candidates = [grid_6x5, grid_7x5_mod, grid_10x3, grid_8x4_mod]
    
    best_points = grid_6x5
    best_score = 0
    
    for pts in candidates:
        if len(pts) != 30:
            continue
        # Check bounds
        valid = all(-30 <= p[0] <= 30 and -30 <= p[1] <= 30 for p in pts)
        if not valid:
            continue
        score = calculate_score(pts)
        if score > best_score:
            best_score = score
            best_points = pts
            
    return best_points
