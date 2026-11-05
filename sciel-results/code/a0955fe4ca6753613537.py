import random
import math
import time

def construct():
    start_time = time.time()
    deadline = 145.0

    def is_perfect_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    def is_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0

    def is_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        def det4(m):
            return (m[0][0]*(m[1][1]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][1] + m[1][3]*m[2][1]*m[3][2] - 
                             m[1][1]*m[2][3]*m[3][2] - m[1][2]*m[2][1]*m[3][3] - m[1][3]*m[2][2]*m[3][1]) -
                    m[0][1]*(m[1][0]*m[2][2]*m[3][3] + m[1][2]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][2] - 
                             m[1][0]*m[2][3]*m[3][2] - m[1][2]*m[2][0]*m[3][3] - m[1][3]*m[2][2]*m[3][0]) +
                    m[0][2]*(m[1][0]*m[2][1]*m[3][3] + m[1][1]*m[2][3]*m[3][0] + m[1][3]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][3]*m[3][1] - m[1][1]*m[2][0]*m[3][3] - m[1][3]*m[2][1]*m[3][0]) -
                    m[0][3]*(m[1][0]*m[2][1]*m[3][2] + m[1][1]*m[2][2]*m[3][0] + m[1][2]*m[2][0]*m[3][1] - 
                             m[1][0]*m[2][2]*m[3][1] - m[1][1]*m[2][0]*m[3][2] - m[1][2]*m[2][1]*m[3][0]))
        
        rows = []
        for p in (p1, p2, p3, p4):
            x, y = p
            rows.append([x*x + y*y, x, y, 1])
        return det4(rows) == 0

    def is_valid_set(points):
        n = len(points)
        if len(set(points)) != n:
            return False
        # Check distances
        for i in range(n):
            for j in range(i+1, n):
                dx = points[i][0] - points[j][0]
                dy = points[i][1] - points[j][1]
                d2 = dx*dx + dy*dy
                if not is_perfect_square(d2):
                    return False
        # Check collinearity
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_collinear(points[i], points[j], points[k]):
                        return False
        # Check concyclicity
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if is_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    def check_new_point(base, new_pt):
        """Fast check if adding new_pt to base maintains validity."""
        x, y = new_pt
        # Check distances to all base points
        for i, (bx, by) in enumerate(base):
            dx = x - bx
            dy = y - by
            d2 = dx*dx + dy*dy
            if not is_perfect_square(d2):
                return False
        # Check collinearity with pairs from base
        n = len(base)
        for i in range(n):
            for j in range(i+1, n):
                if is_collinear(base[i], base[j], new_pt):
                    return False
        # Check concyclicity with triples from base
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_concyclic(base[i], base[j], base[k], new_pt):
                        return False
        return True

    # Seed 5-point set
    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    bases = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (56, 33), (16, 63), (0, 33), (-16, 63)],
        [(0, 0), (7, 24), (15, 8), (24, 7), (32, 24)],
    ]
    
    # Verify bases first
    valid_bases = []
    for base in bases:
        if is_valid_set(base):
            valid_bases.append(base)
            
    if not valid_bases:
        if is_valid_set(seed_points):
            valid_bases.append(seed_points)
        else:
            return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in seed_points]}

    found_6 = None
    search_box = 800
    
    for base in valid_bases:
        if time.time() > deadline - 15.0:
            break
            
        for x in range(-search_box, search_box + 1):
            if time.time() > deadline - 5.0:
                break
            for y in range(-search_box, search_box + 1):
                # Quick check: distance to first point
                dx0 = x - base[0][0]
                dy0 = y - base[0][1]
                d2_0 = dx0*dx0 + dy0*dy0
                if not is_perfect_square(d2_0):
                    continue
                
                # Check distance to other points
                valid_dist = True
                for i in range(1, len(base)):
                    dx = x - base[i][0]
                    dy = y - base[i][1]
                    d2 = dx*dx + dy*dy
                    if not is_perfect_square(d2):
                        valid_dist = False
                        break
                
                if valid_dist:
                    # Fast check for collinearity and concyclicity
                    if check_new_point(base, (x, y)):
                        found_6 = base + [(x, y)]
                        break
            if found_6:
                break
        if found_6:
            break

    if found_6:
        pts = found_6
    else:
        pts = valid_bases[0]

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
