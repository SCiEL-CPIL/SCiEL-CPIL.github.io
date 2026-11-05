import math
import time
from fractions import Fraction

def is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def dist_sq(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

def is_rational_dist(p1, p2, D=1):
    # p1, p2 are (x, y) with x, y in Q or Z
    # We check if (x1-x2)^2 + D(y1-y2)^2 is a rational square
    # Here we assume integer coordinates for simplicity in search
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return is_square(dx*dx + D*dy*dy)

def collinear(p1, p2, p3):
    # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
    return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0

def concyclic(p1, p2, p3, p4):
    # Check if 4 points are concyclic
    # Circle equation: x^2+y^2+Dx+Ey+F=0
    # For integer points, we can use the determinant or solve linear system
    # x1^2+y1^2 + D*x1 + E*y1 + F = 0
    # ...
    # If F=0 (circle through origin), simpler.
    # General case:
    # Determinant of:
    # x^2+y^2  x  y  1
    # is 0
    def det4(m):
        # 4x4 determinant
        return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1])) -
                m[0][1]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])) +
                m[0][2]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])) -
                m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))
    
    m = [
        [p1[0]**2+p1[1]**2, p1[0], p1[1], 1],
        [p2[0]**2+p2[1]**2, p2[0], p2[1], 1],
        [p3[0]**2+p3[1]**2, p3[0], p3[1], 1],
        [p4[0]**2+p4[1]**2, p4[0], p4[1], 1]
    ]
    return det4(m) == 0

def construct():
    start_time = time.time()
    deadline = start_time + 100 # 20s margin
    
    # Seed 5 points
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find a 6th point
    # Search range for x, y
    # Since distances are ~65, new point likely within ~100
    found = False
    for x in range(-100, 101):
        if time.time() > deadline:
            break
        for y in range(-100, 101):
            p_new = (x, y)
            # Check distinct
            if p_new in points:
                continue
            # Check rational distances
            ok = True
            for p in points:
                if not is_rational_dist(p_new, p, 1):
                    ok = False
                    break
            if not ok:
                continue
            # Check no 3 collinear
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    if collinear(points[i], points[j], p_new):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Check no 4 concyclic
            for i in range(len(points)):
                for j in range(i+1, len(points)):
                    for k in range(j+1, len(points)):
                        if concyclic(points[i], points[j], points[k], p_new):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                continue
            
            # Valid 6th point found
            points.append(p_new)
            found = True
            break
        if found:
            break
            
    # If we found 6, try to find 7?
    # Unlikely in time limit.
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
