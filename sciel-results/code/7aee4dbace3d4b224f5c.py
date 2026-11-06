import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin

    # Seed points
    pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Helper to check if a set of points is valid
    def is_valid(points):
        n = len(points)
        # Check distinct
        if len(set(points)) != n:
            return False
        
        # Check no three collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1 = points[i]
                    p2 = points[j]
                    p3 = points[k]
                    # Determinant of matrix [[x1, y1, 1], [x2, y2, 1], [x3, y3, 1]]
                    # x1(y2-y3) - y1(x2-x3) + (x2y3 - x3y2) == 0
                    det = p1[0]*(p2[1]-p3[1]) - p1[1]*(p2[0]-p3[0]) + (p2[0]*p3[1] - p3[0]*p2[1])
                    if det == 0:
                        return False
        
        # Check no four concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1 = points[i]
                        p2 = points[j]
                        p3 = points[k]
                        p4 = points[l]
                        # Four points are concyclic if the determinant of the 
                        # circle equation matrix is 0.
                        # x^2 + y^2 + Ax + By + C = 0
                        # x1^2+y1^2  x1  y1  1
                        # x2^2+y2^2  x2  y2  1
                        # x3^2+y3^2  x3  y3  1
                        # x4^2+y4^2  x4  y4  1
                        # det == 0
                        def det4(m):
                            # m is 4x4
                            # Simple expansion
                            return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1])) -
                                    m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0])) +
                                    m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])) -
                                    m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))
                        
                        m = [
                            [p1[0]**2 + p1[1]**2, p1[0], p1[1], 1],
                            [p2[0]**2 + p2[1]**2, p2[0], p2[1], 1],
                            [p3[0]**2 + p3[1]**2, p3[0], p3[1], 1],
                            [p4[0]**2 + p4[1]**2, p4[0], p4[1], 1]
                        ]
                        if det4(m) == 0:
                            return False
        return True

    # The seed points are already verified by the problem statement/seed code, 
    # but let's be safe.
    if not is_valid(pts):
         # Fallback to a known valid set if seed is broken? Unlikely.
         pass

    best_points = pts
    
    # Try to add a 6th point
    # Search range - expanded significantly
    range_x = range(-500, 501)
    range_y = range(-500, 501)
    
    found_6 = False
    for x in range_x:
        if time.time() - start_time > timeout:
            break
        for y in range_y:
            if time.time() - start_time > timeout:
                break
            new_pt = (x, y)
            # Check if distinct
            if new_pt in pts:
                continue
            
            # Check distances to all 5 points are integers
            valid_dist = True
            for p in pts:
                dx = x - p[0]
                dy = y - p[1]
                d2 = dx*dx + dy*dy
                # Check if d2 is a perfect square
                root = int(math.isqrt(d2))
                if root*root != d2:
                    valid_dist = False
                    break
            
            if not valid_dist:
                continue
            
            # Check general position
            new_set = pts + [new_pt]
            if is_valid(new_set):
                best_points = new_set
                found_6 = True
                # We can break and try to find a 7th, or just return.
                # Let's try to find a 7th if time permits.
                break
        
        if found_6:
            break

    # Try to add a 7th point to the 6-point set if found
    if found_6:
        # Search for 7th
        range_x2 = range(-800, 801)
        range_y2 = range(-800, 801)
        found_7 = False
        for x in range_x2:
            if time.time() - start_time > timeout:
                break
            for y in range_y2:
                if time.time() - start_time > timeout:
                    break
                new_pt = (x, y)
                if new_pt in best_points:
                    continue
                valid_dist = True
                for p in best_points:
                    dx = x - p[0]
                    dy = y - p[1]
                    d2 = dx*dx + dy*dy
                    root = int(math.isqrt(d2))
                    if root*root != d2:
                        valid_dist = False
                        break
                if not valid_dist:
                    continue
                new_set = best_points + [new_pt]
                if is_valid(new_set):
                    best_points = new_set
                    found_7 = True
                    break
            if found_7:
                break

    # Format output
    D = 1
    points_out = [[[x, 1], [y, 1]] for x, y in best_points]
    
    return {'D': D, 'points': points_out}
