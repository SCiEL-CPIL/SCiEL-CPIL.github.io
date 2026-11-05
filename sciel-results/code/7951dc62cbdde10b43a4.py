import math
from fractions import Fraction
from itertools import combinations
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s safety margin (150s limit)
    
    def elapsed():
        return time.time() - start_time
    
    def is_perfect_square(n):
        if n < 0:
            return False
        if n == 0:
            return True
        r = math.isqrt(n)
        return r * r == n
    
    def check_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0
    
    def _exact_det_4x4(p1, p2, p3, p4):
        rows = []
        for p in [p1, p2, p3, p4]:
            x, y = p
            rows.append([x, y, x*x + y*y, 1])
        
        def det3(m):
            return (m[0][0]*(m[1][1]*m[2][2] - m[1][2]*m[2][1])
                   - m[0][1]*(m[1][0]*m[2][2] - m[1][2]*m[2][0])
                   + m[0][2]*(m[1][0]*m[2][1] - m[1][1]*m[2][0]))
        
        def det4(m):
            result = 0
            signs = [1, -1, 1, -1]
            for j in range(4):
                minor = [[m[i][k] for k in range(4) if k != j] for i in range(1, 4)]
                result += signs[j] * m[0][j] * det3(minor)
            return result
        
        return det4(rows)
    
    def all_distances_integer(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                dx = x2 - x1
                dy = y2 - y1
                d2 = dx*dx + dy*dy
                if not is_perfect_square(d2):
                    return False
        return True
    
    def no_three_collinear(points):
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if check_collinear(points[i], points[j], points[k]):
                        return False
        return True
    
    def no_four_concyclic(points):
        n = len(points)
        for combo in combinations(range(n), 4):
            if _exact_det_4x4(points[combo[0]], points[combo[1]], points[combo[2]], points[combo[3]]) == 0:
                return False
        return True
    
    def is_valid(points):
        if not all_distances_integer(points):
            return False
        if not no_three_collinear(points):
            return False
        if not no_four_concyclic(points):
            return False
        return True
    
    # Known 5-point set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    assert is_valid(base_points), "Base set invalid"
    
    best_points = list(base_points)
    best_count = 5
    
    # Try to find 6th point by searching a wider grid
    # Focus on regions where a 6th point might exist
    # The 5 points span x: [-56, 16], y: [0, 63]
    # Try adding points in various regions
    
    search_ranges = [
        # (x_min, x_max, y_min, y_max)
        (-100, 100, -50, 150),   # Wide search
        (-80, 60, -30, 120),     # Medium search
        (-60, 40, 0, 100),       # Focused on existing range
    ]
    
    found_sixth = False
    for x_min, x_max, y_min, y_max in search_ranges:
        if elapsed() > deadline - 20:
            break
        for y in range(y_min, y_max + 1):
            if elapsed() > deadline - 10:
                break
            for x in range(x_min, x_max + 1):
                if elapsed() > deadline - 10:
                    break
                new_p = (x, y)
                if new_p in best_points:
                    continue
                
                # Quick check: distance to each existing point must be integer
                valid_dist = True
                for p in best_points:
                    dx = x - p[0]
                    dy = y - p[1]
                    d2 = dx*dx + dy*dy
                    if not is_perfect_square(d2):
                        valid_dist = False
                        break
                
                if not valid_dist:
                    continue
                
                # Check no three collinear with new point
                collinear = False
                for i in range(len(best_points)):
                    for j in range(i+1, len(best_points)):
                        if check_collinear(best_points[i], best_points[j], new_p):
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    continue
                
                # Check no four concyclic with new point
                concyclic = False
                for combo in combinations(range(len(best_points)), 3):
                    p1, p2, p3 = best_points[combo[0]], best_points[combo[1]], best_points[combo[2]]
                    if _exact_det_4x4(p1, p2, p3, new_p) == 0:
                        concyclic = True
                        break
                if concyclic:
                    continue
                
                # Valid 6th point found
                best_points = best_points + [new_p]
                best_count = 6
                found_sixth = True
                break
        
        if found_sixth:
            break
    
    # If we found 6 points, try to find 7th
    if best_count == 6 and elapsed() < deadline - 30:
        found_seventh = False
        for x_min, x_max, y_min, y_max in search_ranges:
            if elapsed() > deadline - 15:
                break
            for y in range(y_min, y_max + 1):
                if elapsed() > deadline - 10:
                    break
                for x in range(x_min, x_max + 1):
                    if elapsed() > deadline - 10:
                        break
                    new_p = (x, y)
                    if new_p in best_points:
                        continue
                    
                    valid_dist = True
                    for p in best_points:
                        dx = x - p[0]
                        dy = y - p[1]
                        d2 = dx*dx + dy*dy
                        if not is_perfect_square(d2):
                            valid_dist = False
                            break
                    
                    if not valid_dist:
                        continue
                    
                    collinear = False
                    for i in range(len(best_points)):
                        for j in range(i+1, len(best_points)):
                            if check_collinear(best_points[i], best_points[j], new_p):
                                collinear = True
                                break
                        if collinear:
                            break
                    if collinear:
                        continue
                    
                    concyclic = False
                    for combo in combinations(range(len(best_points)), 3):
                        p1, p2, p3 = best_points[combo[0]], best_points[combo[1]], best_points[combo[2]]
                        if _exact_det_4x4(p1, p2, p3, new_p) == 0:
                            concyclic = True
                            break
                    if concyclic:
                        continue
                    
                    best_points = best_points + [new_p]
                    best_count = 7
                    found_seventh = True
                    break
                
                if found_seventh:
                    break
            
            if found_seventh:
                break
    
    # Final verification
    if not is_valid(best_points):
        best_points = list(base_points)
        best_count = 5
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
