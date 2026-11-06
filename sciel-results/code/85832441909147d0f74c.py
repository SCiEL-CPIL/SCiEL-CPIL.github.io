import math
from fractions import Fraction

def construct():
    # Known valid 5-point integral set
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Try to find a 6th point
    # We search for a point (x, y) such that distance to all existing points is integer
    # and no three collinear, no four concyclic (checked later)
    
    best_points = points
    
    # Search range for new point
    # The existing points are in range x: -56..16, y: 0..63
    # Let's search a slightly larger box
    search_range = 100
    
    # To speed up, we can fix x and solve for y, or just brute force small integers
    # Since we need integer distances, (x-xi)^2 + (y-yi)^2 = k^2
    # This is a system of equations.
    # For a small search space, brute force is okay.
    
    for x in range(-search_range, search_range + 1):
        for y in range(-search_range, search_range + 1):
            new_point = (x, y)
            if new_point in points:
                continue
                
            valid = True
            dists = []
            for p in points:
                dx = x - p[0]
                dy = y - p[1]
                d2 = dx*dx + dy*dy
                d = int(math.isqrt(d2))
                if d*d != d2:
                    valid = False
                    break
                dists.append(d)
            
            if not valid:
                continue
                
            # Check collinearity
            new_set = points + [new_point]
            collinear = False
            for i in range(len(new_set)):
                for j in range(i+1, len(new_set)):
                    for k in range(j+1, len(new_set)):
                        p1, p2, p3 = new_set[i], new_set[j], new_set[k]
                        # Cross product
                        if (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0:
                            collinear = True
                            break
                    if collinear:
                        break
                if collinear:
                    break
            if collinear:
                continue
                
            # Check concyclicity (4 points)
            concyclic = False
            for i in range(len(new_set)):
                for j in range(i+1, len(new_set)):
                    for k in range(j+1, len(new_set)):
                        for l in range(k+1, len(new_set)):
                            p1, p2, p3, p4 = new_set[i], new_set[j], new_set[k], new_set[l]
                            # Check if 4 points are concyclic
                            # Determinant of matrix:
                            # x^2+y^2, x, y, 1
                            # If det == 0, they are concyclic
                            det = 0
                            # Use exact arithmetic with integers
                            # | x1^2+y1^2  x1  y1  1 |
                            # | x2^2+y2^2  x2  y2  1 |
                            # | x3^2+y3^2  x3  y3  1 |
                            # | x4^2+y4^2  x4  y4  1 |
                            # We can compute this.
                            # For small ints, it's fine.
                            m = [
                                [p1[0]**2 + p1[1]**2, p1[0], p1[1], 1],
                                [p2[0]**2 + p2[1]**2, p2[0], p2[1], 1],
                                [p3[0]**2 + p3[1]**2, p3[0], p3[1], 1],
                                [p4[0]**2 + p4[1]**2, p4[0], p4[1], 1]
                            ]
                            # Determinant of 4x4
                            # Using Sarrus-like expansion or simple recursion
                            def det4(m):
                                return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]))
                                       - m[0][1]*(m[1][0]*(m[2][2]*m[3][3]-m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]))
                                       + m[0][2]*(m[1][0]*(m[2][1]*m[3][3]-m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3]-m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1]-m[2][1]*m[3][0]))
                                       - m[0][3]*(m[1][0]*(m[2][1]*m[3][2]-m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2]-m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1]-m[2][1]*m[3][0])))
                            if det4(m) == 0:
                                concyclic = True
                                break
                        if concyclic:
                            break
                    if concyclic:
                        break
                if concyclic:
                    break
            if concyclic:
                continue
                
            # If valid, we found a 6-point set
            best_points = new_set
            break
        if len(best_points) == 6:
            break
            
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}
