import random
import math
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    # Target: Maximize points in general position (no 3 collinear, no 4 concyclic)
    # with rational distances.
    # Approach: Generate candidates using Pythagorean triples to ensure rational distance to origin.
    # Then use a greedy/local search to build a set satisfying general position constraints.
    
    def is_square(n):
        if n < 0:
            return False
        r = math.isqrt(n)
        return r * r == n

    def dist_sq(p1, p2):
        # Returns squared distance as Fraction
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def is_rational_distance(p1, p2):
        d2 = dist_sq(p1, p2)
        # d2 is a Fraction. We need sqrt(d2) to be rational.
        # This is equivalent to d2 being a square of a rational number.
        # If d2 = num/den, we need num and den to be perfect squares (after reducing).
        # Actually, sqrt(num/den) = sqrt(num)/sqrt(den). 
        # For this to be rational, num and den must be perfect squares of integers?
        # No. num/den = (a/b)^2 => num*b^2 = den*a^2.
        # It suffices to check if the numerator and denominator of the reduced fraction are perfect squares.
        if d2 == 0:
            return False
        num = d2.numerator
        den = d2.denominator
        if num < 0 or den < 0:
            return False
        sqrt_num = math.isqrt(num)
        sqrt_den = math.isqrt(den)
        return sqrt_num * sqrt_num == num and sqrt_den * sqrt_den == den

    def are_collinear(p1, p2, p3):
        # Use exact arithmetic
        # (x2-x1)(y3-y1) - (y2-y1)(x3-x1) == 0
        val = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        return val == 0

    def are_concyclic(p1, p2, p3, p4):
        # Check determinant of 4x4 matrix
        # | x^2+y^2  x  y  1 |
        # | x^2+y^2  x  y  1 |
        # | x^2+y^2  x  y  1 |
        # | x^2+y^2  x  y  1 |
        # == 0
        
        def dot(u, v):
            return sum(a*b for a, b in zip(u, v))
        
        def cross(u, v):
            return (u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0])
            
        # Vector approach for 4x4 determinant might be complex.
        # Use Cramer's rule or substitution.
        # Circle eq: x^2 + y^2 + Ax + By + C = 0
        # x1^2+y1^2 + A x1 + B y1 + C = 0
        # x2^2+y2^2 + A x2 + B y2 + C = 0
        # x3^2+y3^2 + A x3 + B y3 + C = 0
        
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        # System for A, B, C
        # A x1 + B y1 + C = -(x1^2+y1^2)
        # A x2 + B y2 + C = -(x2^2+y2^2)
        # A x3 + B y3 + C = -(x3^2+y3^2)
        
        # Subtract 1 from 2 and 3
        # A(x2-x1) + B(y2-y1) = -(x2^2+y2^2 - x1^2-y1^2)
        # A(x3-x1) + B(y3-y1) = -(x3^2+y3^2 - x1^2-y1^2)
        
        a11 = x2 - x1
        a12 = y2 - y1
        b1 = -(x2*x2 + y2*y2 - x1*x1 - y1*y1)
        
        a21 = x3 - x1
        a22 = y3 - y1
        b2 = -(x3*x3 + y3*y3 - x1*x1 - y1*y1)
        
        det = a11 * a22 - a12 * a21
        if det == 0:
            # Collinear, not a circle
            return False 
            
        # Solve for A and B
        # A = (b1*a22 - a12*b2) / det
        # B = (a11*b2 - b1*a21) / det
        
        A_num = b1 * a22 - a12 * b2
        B_num = a11 * b2 - b1 * a21
        
        # Check p4:
        x4, y4 = p4
        # x4^2 + y4^2 + A*x4 + B*y4 + C = 0
        # C = -(x1^2+y1^2 + A*x1 + B*y1)
        # So: x4^2 + y4^2 + A*x4 + B*y4 - x1^2 - y1^2 - A*x1 - B*y1 = 0
        # (x4^2 + y4^2 - x1^2 - y1^2) + A(x4 - x1) + B(y4 - y1) = 0
        
        # Substitute A and B:
        # (x4^2 + y4^2 - x1^2 - y1^2) * det + A_num*(x4 - x1) + B_num*(y4 - y1) = 0
        
        val = (x4*x4 + y4*y4 - x1*x1 - y1*y1) * det + A_num*(x4 - x1) + B_num*(y4 - y1)
        return val == 0

    def check_set(points):
        n = len(points)
        # Check pairwise distances are rational
        for i in range(n):
            for j in range(i+1, n):
                if not is_rational_distance(points[i], points[j]):
                    return False
        
        # Check no 3 collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if are_collinear(points[i], points[j], points[k]):
                        return False
        
        # Check no 4 concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        if are_concyclic(points[i], points[j], points[k], points[l]):
                            return False
        return True

    # Generate candidates using Pythagorean triples
    # Point (x, y) = (k * (u^2-v^2)/(u^2+v^2), k * 2uv/(u^2+v^2))
    # Distance to origin is k.
    # We will use Fractions for x, y to be exact.
    
    candidates = set()
    # Add origin
    candidates.add((Fraction(0,1), Fraction(0,1)))
    
    # Generate points with small u, v, k
    # To keep numbers manageable, we limit u, v, k
    max_uv = 20
    max_k = 10
    
    for u in range(1, max_uv + 1):
        for v in range(1, u):
            if (u - v) % 2 == 0: # Ensure primitive or just valid? Any valid triple works.
                # Actually, any u>v>0 generates a rational point on unit circle.
                pass
            
            u2 = u * u
            v2 = v * v
            denom = u2 + v2
            
            # Base point on unit circle
            x_base = Fraction(u2 - v2, denom)
            y_base = Fraction(2 * u * v, denom)
            
            # Scale by k
            for k in range(1, max_k + 1):
                x = x_base * k
                y = y_base * k
                candidates.add((x, y))
                # Also add negative and swapped to cover all quadrants/symmetries?
                # The formula covers Q1. We can reflect.
                candidates.add((x, -y))
                candidates.add((-x, y))
                candidates.add((-x, -y))
                # Swap x and y (reflection across y=x)
                candidates.add((y, x))
                candidates.add((y, -x))
                candidates.add((-y, x))
                candidates.add((-y, -x))
                
    # Convert to list
    cand_list = list(candidates)
    random.seed(42)
    random.shuffle(cand_list)
    
    # Greedy construction
    # Start with a known good set or just try to build from scratch?
    # Building from scratch with origin is good.
    # But we need to ensure general position.
    
    best_set = []
    current_set = [(Fraction(0,1), Fraction(0,1))]
    
    # We'll iterate through candidates and try to add them.
    # To speed up, we can pre-filter candidates that are at rational distance to current_set.
    
    for p in cand_list:
        if p in current_set:
            continue
            
        # Check rational distance to all in current_set
        valid_dist = True
        for q in current_set:
            if not is_rational_distance(p, q):
                valid_dist = False
                break
        
        if not valid_dist:
            continue
            
        # Check general position
        new_set = current_set + [p]
        if check_set(new_set):
            current_set = new_set
            if len(current_set) > len(best_set):
                best_set = current_set[:]
    
    # If we found a good set, try to improve it with local search?
    # Given time limits, a single greedy pass might be enough if the pool is good.
    # Let's try a second pass with a different random seed or order if we have time.
    
    # Format output
    points_out = []
    for x, y in best_set:
        points_out.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
        
    return {'D': 1, 'points': points_out}

# EVOLVE-BLOCK-END
