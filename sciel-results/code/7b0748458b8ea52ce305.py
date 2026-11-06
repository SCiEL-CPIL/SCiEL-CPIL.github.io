import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    def is_int_dist_sq(dx, dy):
        d2 = dx*dx + dy*dy
        r = math.isqrt(d2)
        return r*r == d2

    def is_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0])

    def is_concyclic(p1, p2, p3, p4):
        x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
        A1 = x2*x2 + y2*y2 - x1*x1 - y1*y1
        B1 = x2 - x1
        C1 = y2 - y1
        A2 = x3*x3 + y3*y3 - x1*x1 - y1*y1
        B2 = x3 - x1
        C2 = y3 - y1
        det = B1*C2 - B2*C1
        if det == 0:
            return False
        D = (A1*C2 - A2*C1) // det
        E = (B1*A2 - B2*A1) // det
        F = -(x1*x1 + y1*y1 + D*x1 + E*y1)
        return x4*x4 + y4*y4 + D*x4 + E*y4 + F == 0

    def validate_points(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                if not is_int_dist_sq(dx, dy):
                    return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        return False
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if is_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def try_extend(points, deadline, max_radius=200):
        n = len(points)
        cx = sum(p[0] for p in points) // n
        cy = sum(p[1] for p in points) // n
        
        # Try points in a structured pattern
        # Use Pythagorean-based offsets
        pythagorean = [(0,0),(1,0),(0,1),(1,1),(3,4),(4,3),(5,12),(12,5),
                       (8,15),(15,8),(7,24),(24,7),(20,21),(21,20),(9,40),(40,9),
                       (12,35),(35,12),(11,60),(60,11),(28,45),(45,28),(33,56),(56,33),
                       (16,63),(63,16),(48,55),(55,48),(13,84),(84,13),(36,77),(77,36)]
        
        # Generate candidate positions
        candidates = set()
        for dx, dy in pythagorean:
            for sign_x in [-1, 1]:
                for sign_y in [-1, 1]:
                    candidates.add((cx + sign_x*dx, cy + sign_y*dy))
        
        # Also try some random-looking positions
        import random
        random.seed(42)
        for _ in range(500):
            x = cx + random.randint(-150, 150)
            y = cy + random.randint(-150, 150)
            candidates.add((x, y))
        
        # Sort candidates by distance from centroid for deterministic order
        candidates = sorted(candidates, key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
        
        for p_new in candidates:
            if time.time() > deadline:
                break
            
            if p_new in points:
                continue
            
            # Distance check
            valid_dist = True
            for p in points:
                dx = p_new[0] - p[0]
                dy = p_new[1] - p[1]
                if not is_int_dist_sq(dx, dy):
                    valid_dist = False
                    break
            if not valid_dist:
                continue
            
            # Collinearity
            valid_col = True
            for i in range(n):
                for j in range(i+1, n):
                    if is_collinear(points[i], points[j], p_new):
                        valid_col = False
                        break
                if not valid_col:
                    break
            if not valid_col:
                continue
            
            # Concyclicity
            valid_con = True
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        if is_concyclic(points[i], points[j], points[k], p_new):
                            valid_con = False
                            break
                        if not valid_con:
                            break
                    if not valid_con:
                        break
                if not valid_con:
                    break
            if not valid_con:
                continue
            
            return p_new
        
        return None

    # Known valid 5-point configuration
    seed = [(0, 0), (3, 4), (0, 5), (3, 1), (5, 3)]
    
    # Verify seed
    if not validate_points(seed):
        # Try another seed
        seed = [(0, 0), (4, 3), (0, 5), (4, 8), (9, 1)]
    
    best_points = list(seed)
    
    # Try to extend
    for _ in range(7):
        if time.time() > deadline:
            break
        new_p = try_extend(best_points, deadline)
        if new_p is None:
            break
        best_points.append(new_p)
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
