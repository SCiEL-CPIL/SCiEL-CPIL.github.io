import math
import random
import time
from fractions import Fraction

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _check_pairwise_distances_rational(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x1 - x2
            dy = y1 - y2
            dist_sq = dx * dx + D * dy * dy
            if not _is_square(dist_sq):
                return False
    return True

def _check_no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                # (x2-x1)(y3-y1) == (y2-y1)(x3-x1)
                if (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1):
                    return False
    return True

def _check_no_four_concyclic(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    a1 = x1*x1 + D*y1*y1
                    a2 = x2*x2 + D*y2*y2
                    a3 = x3*x3 + D*y3*y3
                    a4 = x4*x4 + D*y4*y4
                    
                    # 4x4 determinant
                    # | a1 x1 y1 1 |
                    # | a2 x2 y2 1 |
                    # | a3 x3 y3 1 |
                    # | a4 x4 y4 1 |
                    # Using expansion
                    def det4(m):
                        return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3] - m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3] - m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2] - m[2][2]*m[3][1])) -
                                m[0][1]*(m[1][0]*(m[2][2]*m[3][3] - m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3] - m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2] - m[2][2]*m[3][0])) +
                                m[0][2]*(m[1][0]*(m[2][1]*m[3][3] - m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3] - m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1] - m[2][1]*m[3][0])) -
                                m[0][3]*(m[1][0]*(m[2][1]*m[3][2] - m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2] - m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1] - m[2][1]*m[3][0])))
                    
                    m = [[a1, x1, y1, 1], [a2, x2, y2, 1], [a3, x3, y3, 1], [a4, x4, y4, 1]]
                    if det4(m) == 0:
                        return False
    return True

def _is_valid_candidate(pt, existing_points, D):
    x, y = pt
    # Check distinctness
    for ex in existing_points:
        if ex == pt:
            return False
    
    # Check distances to all existing points
    for ex in existing_points:
        ex_x, ex_y = ex
        dx = x - ex_x
        dy = y - ex_y
        d2 = dx*dx + D*dy*dy
        if not _is_square(d2):
            return False
            
    # Check collinearity with any pair of existing points
    n = len(existing_points)
    for i in range(n):
        for j in range(i+1, n):
            x1, y1 = existing_points[i]
            x2, y2 = existing_points[j]
            # (x2-x1)(y-y1) == (y2-y1)(x-x1)
            if (x2 - x1) * (y - y1) == (y2 - y1) * (x - x1):
                return False
                
    # Check concyclicity with any triple of existing points
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                x1, y1 = existing_points[i]
                x2, y2 = existing_points[j]
                x3, y3 = existing_points[k]
                a1 = x1*x1 + D*y1*y1
                a2 = x2*x2 + D*y2*y2
                a3 = x3*x3 + D*y3*y3
                a4 = x*x + D*y*y
                
                # Det of 4x4 matrix
                def det4(m):
                    return (m[0][0]*(m[1][1]*(m[2][2]*m[3][3] - m[2][3]*m[3][2]) - m[1][2]*(m[2][1]*m[3][3] - m[2][3]*m[3][1]) + m[1][3]*(m[2][1]*m[3][2] - m[2][2]*m[3][1])) -
                            m[0][1]*(m[1][0]*(m[2][2]*m[3][3] - m[2][3]*m[3][2]) - m[1][2]*(m[2][0]*m[3][3] - m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][2] - m[2][2]*m[3][0])) +
                            m[0][2]*(m[1][0]*(m[2][1]*m[3][3] - m[2][3]*m[3][1]) - m[1][1]*(m[2][0]*m[3][3] - m[2][3]*m[3][0]) + m[1][3]*(m[2][0]*m[3][1] - m[2][1]*m[3][0])) -
                            m[0][3]*(m[1][0]*(m[2][1]*m[3][2] - m[2][2]*m[3][1]) - m[1][1]*(m[2][0]*m[3][2] - m[2][2]*m[3][0]) + m[1][2]*(m[2][0]*m[3][1] - m[2][1]*m[3][0])))
                            
                m = [[a1, x1, y1, 1], [a2, x2, y2, 1], [a3, x3, y3, 1], [a4, x, y, 1]]
                if det4(m) == 0:
                    return False
                    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    D = 1
    # Seed 5 points
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    # Verify seed
    if not (_check_pairwise_distances_rational(points, D) and _check_no_three_collinear(points) and _check_no_four_concyclic(points, D)):
        # Fallback to even simpler 3 points if seed is bad (shouldn't happen)
        points = [(0,0), (3,0), (0,4)]
        
    best_points = points
    
    # Strategy: Hilbert 90 parameterization to generate points on circles (in Q) 
    # that have rational distances to origin.
    # Distance from origin (0,0) to (x,y) is sqrt(x^2 + D*y^2).
    # We want this to be rational.
    # Let x^2 + D*y^2 = r^2.
    # This is a conic. We can parameterize solutions.
    # For D=1, x^2 + y^2 = r^2.
    # Param: x = r * (1-t^2)/(1+t^2), y = r * 2t/(1+t^2) for rational t.
    # But we need integer coordinates for efficiency? The task allows rational x,y.
    # However, integer coordinates are easier to check with _is_square.
    # Pythagorean triples: (3,4,5), (5,12,13), etc.
    # We can generate points (x,y) such that x^2+y^2 is a square.
    
    # Generate a pool of candidate points
    candidates = set()
    # Add some known Pythagorean triples scaled/shifted?
    # Actually, just generate points with integer coords that are rationally distant from (0,0).
    # And also from other seed points?
    # The condition is pairwise rational distances.
    # If we pick points from a set where all pairwise distances are rational, we are good.
    # The seed points are already such a set.
    # We can try to find more points that are rationally distant from ALL seed points.
    
    # Let's generate candidates in a larger box.
    # To ensure rational distance to (0,0), we need x^2+y^2 to be a square.
    # Let's iterate over y and solve for x? Or just iterate x,y and check.
    
    # Efficient candidate generation:
    # Iterate over possible distances r (integer).
    # For each r, find integer solutions to x^2 + y^2 = r^2.
    # This gives points on circles centered at origin.
    # But we need rational distance to OTHER points too.
    
    # Let's just do a random search first, but smarter.
    # Random search in a box, check validity.
    
    # To improve search, we can use the fact that if P and Q are valid,
    # and we find R such that dist(P,R) and dist(Q,R) are rational,
    # we still need to check all pairs.
    
    # Let's try to generate candidates that are "close" to the existing points in some sense,
    # or just random.
    
    # A better approach:
    # Generate random Pythagorean triples (a,b,c) -> point (a,b).
    # Check if it's valid with existing points.
    
    def generate_pythagorean_point():
        # Generate random m, n
        m = random.randint(2, 100)
        n = random.randint(1, m-1)
        if (m-n) % 2 == 0: # Ensure primitive? Not strictly necessary, but good for variety
            pass
        a = m*m - n*n
        b = 2*m*n
        c = m*m + n*n
        # Randomly swap a, b and random sign
        if random.random() < 0.5:
            a, b = b, a
        a = a if random.random() < 0.5 else -a
        b = b if random.random() < 0.5 else -b
        return (a, b)
        
    # Also try random integer points, but filter by distance to (0,0) first?
    # No, distance to (0,0) is just one constraint.
    # Random integer points are unlikely to satisfy distance to (0,0) AND (-56,33).
    
    # So, generate points that are rationally distant from (0,0) is a good filter.
    # Then check against the rest.
    
    attempts = 0
    while time.time() < deadline and len(points) < 8:
        attempts += 1
        
        # Generate candidate
        # Mix of Pythagorean points (rational dist to origin) and random
        if random.random() < 0.8:
            cand = generate_pythagorean_point()
        else:
            cand = (random.randint(-500, 500), random.randint(-500, 500))
            
        if cand in points:
            continue
            
        # Quick check: distance to (0,0)
        x, y = cand
        if not _is_square(x*x + D*y*y):
            continue
            
        # Check full validity
        if _is_valid_candidate(cand, points, D):
            points.append(cand)
            best_points = points[:]
            
    # Return the best points found
    return {'D': D, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}

