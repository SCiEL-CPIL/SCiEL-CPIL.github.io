import math
import random
import time
from fractions import Fraction

def is_square(n):
    if n < 0:
        return False
    r = int(math.isqrt(n))
    return r * r == n

def check_collinear(p1, p2, p3):
    # p = (x_num, x_den, y_num, y_den)
    x1, x2, x3 = Fraction(p1[0], p1[1]), Fraction(p2[0], p2[1]), Fraction(p3[0], p3[1])
    y1, y2, y3 = Fraction(p1[2], p1[3]), Fraction(p2[2], p2[3]), Fraction(p3[2], p3[3])
    # Area of triangle
    area = x1*(y2-y3) - x2*(y1-y3) + x3*(y1-y2)
    return area == 0

def check_concyclic(p1, p2, p3, p4):
    # Determinant of the 4x4 matrix for circle equation
    # x^2 + y^2 + Dx + Ey + F = 0
    # We use the property that 4 points are concyclic iff the determinant is 0
    # For exact arithmetic, we can use the cross product of vectors in homogeneous coordinates
    # Or simply check if the perpendicular bisectors intersect at a center equidistant from all 4.
    
    # Simpler: Check if the angle subtended by a chord is the same.
    # Or use the determinant formula:
    # | x1^2+y1^2, x1, y1, 1 |
    # | x2^2+y2^2, x2, y2, 1 |
    # | x3^2+y3^2, x3, y3, 1 | = 0
    # | x4^2+y4^2, x4, y4, 1 |
    
    # Note: This works for Euclidean metric.
    # Our metric is (dx)^2 + D(dy)^2.
    # Concyclicity in the D-metric?
    # The task says "no four concyclic". This usually refers to the standard Euclidean circle in the (x, y*sqrt(D)) plane.
    # So we check standard concyclicity of the points (x, y*sqrt(D)).
    
    x1 = Fraction(p1[0], p1[1])
    y1 = Fraction(p1[2], p1[3]) * math.sqrt(p1[4]) # Wait, D is global.
    
    # Let's pass D to the checkers or make them closure.
    pass

def construct():
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Seed set of 5 points for D=1
    # (0,0), (-56,33), (-16,63), (0,33), (16,63)
    seed_points = [
        [0, 1, 0, 1],
        [-56, 1, 33, 1],
        [-16, 1, 63, 1],
        [0, 1, 33, 1],
        [16, 1, 63, 1]
    ]
    
    best_result = {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in [(0,0), (-56,33), (-16,63), (0,33), (16,63)]]}
    
    # Try to find 6 points for D=1
    # We will try to add a point to the seed set.
    # This is a specific search.
    
    # If we can't find 6, we return 5.
    
    # For now, return the seed.
    # A more sophisticated search would be implemented here.
    
    return best_result
