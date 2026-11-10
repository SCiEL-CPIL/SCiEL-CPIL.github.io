import time
from fractions import Fraction
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Helper to check if a number is a perfect square
    def is_square(n):
        if n < 0:
            return False
        r = int(math.isqrt(n))
        return r * r == n

    # Helper to check if a rational number is a square of a rational
    # If we use integer coords and D=1, d^2 is integer.
    # We need d to be rational. d^2 integer => d rational iff d^2 is perfect square integer.
    
    def check_distance(p1, p2, D=1):
        x1, y1 = p1
        x2, y2 = p2
        # d^2 = (x2-x1)^2 + D*(y2-y1)^2
        # If D=1 and coords are int, d^2 is int.
        dx = x2 - x1
        dy = y2 - y1
        d2 = dx*dx + D*dy*dy
        if D == 1:
            return is_square(d2)
        else:
            # General case: d^2 = A + B*sqrt(D) is not rational unless B=0? 
            # Wait. Distance squared is (dx)^2 + D(dy)^2.
            # If D is not a square, this is in Q(sqrt(D)).
            # For the distance d to be rational, d^2 must be in Q.
            # So D*(dy)^2 must be in Q. Since D is squarefree, dy must be 0?
            # No. d^2 = dx^2 + D dy^2.
            # If d is rational, d^2 is rational.
            # dx^2 is rational. D dy^2 must be rational.
            # If D is not a square, dy^2 must be 0? No, dy is rational.
            # D dy^2 is in Q(sqrt(D)).
            # The only way a non-zero element of Q(sqrt(D)) is in Q is if it is 0?
            # No. a + b sqrt(D) is in Q iff b=0.
            # Here d^2 = dx^2 + (dy^2) * sqrt(D)^2? No.
            # d^2 = (x2-x1)^2 + D(y2-y1)^2.
            # This is a rational number if D is a square.
            # If D is not a square, d^2 is in Q(sqrt(D)).
            # For d to be rational, d^2 must be rational.
            # So we need the coefficient of sqrt(D) to be 0?
            # Wait. d^2 is just a number.
            # If D is not a square, d^2 is irrational unless dy=0.
            # If dy != 0, d^2 = A + B sqrt(D) with B != 0.
            # This is irrational.
            # So d = sqrt(A + B sqrt(D)).
            # Can this be rational?
            # If d is rational, d^2 is rational.
            # So A + B sqrt(D) must be rational.
            # This implies B = 0, so dy = 0.
            # So if D is not a square, all points must have the same y-coordinate?
            # That would make them collinear.
            # So D MUST be a square?
            # The task says D is squarefree.
            # So D=1 is the only option for non-collinear points with rational distances?
            # Let's re-read carefully.
            # "x, y*sqrt(D) with rational x, y".
            # Distance squared = (x2-x1)^2 + D(y2-y1)^2.
            # If D is not a square, this sum is in Q(sqrt(D)).
            # For the distance to be rational, the squared distance must be a square of a rational.
            # If d is rational, d^2 is rational.
            # So (x2-x1)^2 + D(y2-y1)^2 must be rational.
            # Since (x2-x1)^2 is rational, D(y2-y1)^2 must be rational.
            # Let y1, y2 be rational. Let y1 = a/b, y2 = c/d.
            # D * (y2-y1)^2 is in Q(sqrt(D)) only if we consider the field.
            # But (y2-y1) is rational, so (y2-y1)^2 is rational.
            # So D * (rational)^2 is in Q(sqrt(D)).
            # For this to be in Q, the coefficient of sqrt(D) must be 0.
            # This implies (y2-y1)^2 * D = 0?
            # No. D * q^2 is just a number.
            # If D=2, y1=0, y2=1. d^2 = dx^2 + 2.
            # If dx=1, d^2=3. d=sqrt(3) irrational.
            # If dx=sqrt(2)? No, x is rational.
            # So dx^2 is rational.
            # So d^2 = Rational + D * Rational.
            # This is a number of the form A + B*D? No.
            # d^2 = A + D*B where A, B are rational.
            # This is a rational number!
            # Wait. D is an integer. B is rational. D*B is rational.
            # So d^2 is always rational!
            # My previous thought was wrong. I was confusing d^2 with d.
            # d^2 is rational.
            # We need d to be rational.
            # So d^2 must be a perfect square of a rational number.
            # Let d^2 = k/l. We need k/l to be a square of a rational.
            # i.e., k and l must be perfect squares (in reduced form).
            
            return True # Placeholder, need exact check

    # Exact check for rational distance with D
    def is_rational_distance(p1, p2, D):
        x1, y1 = p1
        x2, y2 = p2
        # x, y are Fractions
        dx = x2 - x1
        dy = y2 - y1
        d2 = dx*dx + D * dy * dy
        # d2 is a Fraction
        # Check if d2 is a square of a rational
        # d2 = n/d. We need n and d to be perfect squares.
        n = d2.numerator
        d = d2.denominator
        if n < 0 or d < 0:
            return False
        return is_square(n) and is_square(d)

    def is_collinear(p1, p2, p3):
        # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
        return (p2[0]-p1[0])*(p3[1]-p1[1]) - (p2[1]-p1[1])*(p3[0]-p1[0]) == 0

    def is_concyclic(p1, p2, p3, p4):
        # Check determinant
        # | x^2+y^2, x, y, 1 |
        # Use exact arithmetic
        # Circle equation: x^2 + y^2 + Ax + By + C = 0
        # 3 points define a circle. Check if 4th lies on it.
        # Or use the determinant of the 4x4 matrix.
        # For simplicity and speed, use the property:
        # Points are concyclic iff the angle subtended by a chord is the same.
        # Or: (p1-p3)(p2-p4) dot (p1-p4)(p2-p3) ... no.
        # Let's use the determinant method.
        # M = [[x1^2+y1^2, x1, y1, 1], ...]
        # det(M) == 0
        # Note: y^2 in the circle equation for (x, y*sqrt(D)) is y^2 * D.
        # So the term is x^2 + D*y^2.
        
        def val(p):
            return p[0]*p[0] + D*p[1]*p[1]
            
        # Construct 4x4 matrix
        # We can compute determinant exactly.
        # To save time, we can check if p4 lies on circle defined by p1,p2,p3.
        # If p1,p2,p3 are collinear, they don't define a unique circle. 
        # But "no three collinear" is a constraint we enforce.
        # So we assume non-collinear.
        
        # Equation: x^2 + D y^2 + Ax + By + C = 0
        # A x1 + B y1 + C = -(x1^2 + D y1^2)
        # Solve for A, B, C using p1, p2, p3.
        # Then check p4.
        
        # Linear system:
        # x1 A + y1 B + C = -v1
        # x2 A + y2 B + C = -v2
        # x3 A + y3 B + C = -v3
        
        # Subtract:
        # (x2-x1) A + (y2-y1) B = -(v2-v1)
        # (x3-x1) A + (y3-y1) B = -(v3-v1)
        
        a11 = x2 - x1
        a12 = y2 - y1
        a21 = x3 - x1
        a22 = y3 - y1
        b1 = -(v2 - v1)
        b2 = -(v3 - v1)
        
        det = a11 * a22 - a12 * a21
        if det == 0:
            return False # Collinear, handled elsewhere
        
        A = (b1 * a22 - a12 * b2) / det
        B = (a11 * b2 - b1 * a21) / det
        C = -v1 - A * x1 - B * y1
        
        # Check p4
        # x4^2 + D y4^2 + A x4 + B y4 + C == 0
        return val(p4) + A * x4 + B * y4 + C == 0

    def check_set(points):
        n = len(points)
        # Check pairwise distances
        for i in range(n):
            for j in range(i+1, n):
                if not is_rational_distance(points[i], points[j], D):
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

    # Initial set
    D = 1
    # Convert to Fractions
    init_points = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63))
    ]
    
    best_points = init_points
    best_size = len(best_points)
    
    # Search for more points
    # We will try to add points from a grid
    # To speed up, we only check against the current set
    # We will use a random walk / local search
    
    # Candidate generation
    # We'll search in a box around the points
    # x in [-100, 100], y in [-100, 100]
    # This is 200x200 = 40000 points.
    # Checking 5 distances + 5 collinear + 5 concyclic is fast.
    # 40000 * 5 * (some ops) is small.
    
    # However, we might need larger coordinates.
    # Let's try a few different grid sizes.
    
    def try_add_point(current_set, candidates):
        for p in candidates:
            # Check if p is already in set
            if any(p == q for q in current_set):
                continue
            # Check distances
            ok = True
            for q in current_set:
                if not is_rational_distance(p, q, D):
                    ok = False
                    break
            if not ok:
                continue
            # Check collinear
            for i in range(len(current_set)):
                for j in range(i+1, len(current_set)):
                    if is_collinear(current_set[i], current_set[j], p):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Check concyclic
            for i in range(len(current_set)):
                for j in range(i+1, len(current_set)):
                    for k in range(j+1, len(current_set)):
                        if is_concyclic(current_set[i], current_set[j], current_set[k], p):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                new_set = current_set + [p]
                return new_set
        return None

    # Generate candidates
    # We will try to find a 6th point
    # Then 7th, etc.
    
    # To be efficient, we generate a list of candidates once
    # and try to add them one by one.
    
    # But adding a point changes the constraints for the next point.
    # So we do:
    # current = best_points
    # for size in range(best_size+1, 13):
    #    found = False
    #    for p in candidates:
    #       if can_add(current, p):
    #           current.append(p)
    #           found = True
    #           break
    #    if not found: break
    
    # This greedy approach might fail if the first added point blocks better future points.
    # But it's a good heuristic.
    
    # Let's create a diverse set of candidates.
    candidates = []
    # Grid search
    for x in range(-100, 101):
        for y in range(-100, 101):
            candidates.append((Fraction(x), Fraction(y)))
    
    # Shuffle to avoid bias
    random.shuffle(candidates)
    
    current = list(best_points)
    
    while time.time() < deadline:
        added = False
        for p in candidates:
            if p in current:
                continue
            # Check if p can be added to current
            ok = True
            for q in current:
                if not is_rational_distance(p, q, D):
                    ok = False
                    break
            if not ok:
                continue
            # Collinear
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    if is_collinear(current[i], current[j], p):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            # Concyclic
            for i in range(len(current)):
                for j in range(i+1, len(current)):
                    for k in range(j+1, len(current)):
                        if is_concyclic(current[i], current[j], current[k], p):
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if ok:
                current.append(p)
                added = True
                if len(current) > best_size:
                    best_points = list(current)
                    best_size = len(current)
                break # Restart candidate loop with new set
        
        if not added:
            # Try random perturbation? Or break?
            # If we can't add any more, break.
            break

    # Return best found
    return {'D': D, 'points': [[[x.numerator, x.denominator], [y.numerator, y.denominator]] for x, y in best_points]}
