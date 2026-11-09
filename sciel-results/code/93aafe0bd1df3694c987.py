import time
import math
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def is_collinear(p1, p2, p3):
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]) == 0

    def is_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        x2p = x2 - x1
        y2p = y2 - y1
        x3p = x3 - x1
        y3p = y3 - y1
        x4p = x4 - x1
        y4p = y4 - y1
        
        d2 = x2p*x2p + y2p*y2p
        d3 = x3p*x3p + y3p*y3p
        d4 = x4p*x4p + y4p*y4p
        
        det = d2 * (x3p * y4p - x4p * y3p) - x2p * (d3 * y4p - d4 * y3p) + y2p * (d3 * x4p - d4 * x3p)
        return det == 0

    def can_add_point(points, new_point):
        n = len(points)
        
        # Check distances
        for i in range(n):
            dx = points[i][0] - new_point[0]
            dy = points[i][1] - new_point[1]
            d2 = dx*dx + dy*dy
            if not is_square(d2):
                return False
        
        # Check collinearity
        for i in range(n):
            for j in range(i+1, n):
                if is_collinear(points[i], points[j], new_point):
                    return False
        
        # Check concyclicity
        for combo in combinations(range(n), 3):
            if is_concyclic(points[combo[0]], points[combo[1]], points[combo[2]], new_point):
                return False
                
        return True

    def try_add_point(points, max_coord=200):
        cx = sum(p[0] for p in points) // len(points)
        cy = sum(p[1] for p in points) // len(points)
        
        # Randomize search order
        x_range = list(range(cx - max_coord, cx + max_coord + 1))
        y_range = list(range(cy - max_coord, cy + max_coord + 1))
        random.shuffle(x_range)
        random.shuffle(y_range)
        
        for x in x_range:
            for y in y_range:
                new_point = (x, y)
                if new_point in points:
                    continue
                if can_add_point(points, new_point):
                    return new_point
        return None

    # Known 5-point set
    known_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best_points = known_5
    best_size = 5
    
    # Try to grow
    current_points = known_5[:]
    
    # Try to add a 6th point
    while time.time() < deadline:
        new_point = try_add_point(current_points, max_coord=400)
        if new_point:
            current_points.append(new_point)
            best_points = current_points[:]
            best_size = len(current_points)
            # Try to add another
            # But if we add one, the search space for the next is huge.
            # Let's try to add another immediately.
            continue
        else:
            # Stuck. Break.
            break
    
    # If we didn't find 6, maybe try a different 5-point set?
    # Or just return the best we have.
    
    points_out = []
    for x, y in best_points:
        points_out.append([[[x, 1], [y, 1]]])
        
    return {'D': 1, 'points': points_out}
