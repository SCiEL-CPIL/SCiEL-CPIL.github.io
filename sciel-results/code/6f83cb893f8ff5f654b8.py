import math
import random
import time
from itertools import combinations
from fractions import Fraction

def is_square(n):
    if n < 0:
        return False
    if isinstance(n, Fraction):
        n = n.numerator // n.denominator  # won't work, need proper check
    r = math.isqrt(n) if isinstance(n, int) else math.isqrt(n.numerator)
    if isinstance(n, Fraction):
        rn = math.isqrt(n.numerator)
        rd = math.isqrt(n.denominator)
        return rn * rn == n.numerator and rd * rd == n.denominator
    return r * r == n

def dist_sq_frac(p1, p2):
    """Squared distance between two points with Fraction coordinates."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def is_rational_dist(p1, p2):
    """Check if distance between two Fraction-coordinate points is rational."""
    d2 = dist_sq_frac(p1, p2)
    if d2 <= 0:
        return False
    num = d2.numerator
    den = d2.denominator
    rn = math.isqrt(num)
    rd = math.isqrt(den)
    return rn * rn == num and rd * rd == den

def is_collinear_frac(p1, p2, p3):
    """Check collinearity with Fraction coordinates."""
    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    return cross == 0

def is_concyclic_frac(p1, p2, p3, p4):
    """Check if 4 points with Fraction coords are concyclic."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    r1 = x1*x1 + y1*y1
    r2 = x2*x2 + y2*y2
    r3 = x3*x3 + y3*y3
    r4 = x4*x4 + y4*y4
    
    A, B, C, Dv = r1, x1, y1, Fraction(1)
    E, F, G, H = r2, x2, y2, Fraction(1)
    I, J, K, L = r3, x3, y3, Fraction(1)
    M, N, O, P = r4, x4, y4, Fraction(1)
    
    det = (
        A * (F * K * P - F * L * O - G * J * P + G * L * N + H * J * O - H * K * N)
        - B * (E * K * P - E * L * O - G * I * P + G * L * M + H * I * O - H * K * M)
        + C * (E * J * P - E * L * N - F * I * P + F * L * M + H * I * N - H * J * M)
        - Dv * (E * J * O - E * K * N - F * I * O + F * K * M + G * I * N - G * J * M)
    )
    return det == 0

def check_validity_frac(points):
    """Validate a set of Fraction-coordinate points."""
    n = len(points)
    # Check distinct
    for i in range(n):
        for j in range(i+1, n):
            if points[i][0] == points[j][0] and points[i][1] == points[j][1]:
                return False
    # Check distances
    for i in range(n):
        for j in range(i+1, n):
            if not is_rational_dist(points[i], points[j]):
                return False
    # Check collinearity
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if is_collinear_frac(points[i], points[j], points[k]):
                    return False
    # Check concyclicity
    for quad in combinations(range(n), 4):
        if is_concyclic_frac(points[quad[0]], points[quad[1]], points[quad[2]], points[quad[3]]):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # Known 5-point set (D=1, integer coords)
    known5 = [(Fraction(0), Fraction(0)),
              (Fraction(-56), Fraction(33)),
              (Fraction(-16), Fraction(63)),
              (Fraction(0), Fraction(33)),
              (Fraction(16), Fraction(63))]
    
    best_points = known5
    best_size = 5
    
    # Verify the known set
    if check_validity_frac(known5):
        pass  # Good
    else:
        # Fallback: use a simpler known set
        best_points = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4)), (Fraction(6), Fraction(8))]
        best_size = 3
    
    # Strategy: Try to grow the set by adding points one at a time
    # For each candidate position, check all constraints with existing points
    
    def try_add_point(existing, candidate):
        """Check if candidate can be added to existing set."""
        # Check distinct
        for p in existing:
            if p[0] == candidate[0] and p[1] == candidate[1]:
                return False
        # Check distances
        for p in existing:
            if not is_rational_dist(p, candidate):
                return False
        # Check collinearity with pairs
        for i in range(len(existing)):
            for j in range(i+1, len(existing)):
                if is_collinear_frac(existing[i], existing[j], candidate):
                    return False
        # Check concyclicity with triples
        for i in range(len(existing)):
            for j in range(i+1, len(existing)):
                for k in range(j+1, len(existing)):
                    if is_concyclic_frac(existing[i], existing[j], existing[k], candidate):
                        return False
        return True
    
    # Try to find a 6th point near the existing 5
    # Use a grid search around the convex hull
    if time.time() < deadline:
        # Try integer points in a reasonable range
        found6 = False
        for restart in range(50):
            if time.time() > deadline:
                break
            
            # Try random integer points
            for attempt in range(2000):
                if time.time() > deadline:
                    break
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                cand = (Fraction(x), Fraction(y))
                if try_add_point(best_points, cand):
                    new_set = best_points + [cand]
                    if check_validity_frac(new_set):
                        best_points = new_set
                        best_size = 6
                        found6 = True
                        break
            if found6:
                break
        
        # If we found 6, try to find a 7th
        if best_size == 6 and time.time() < deadline:
            found7 = False
            for attempt in range(5000):
                if time.time() > deadline:
                    break
                x = random.randint(-100, 100)
                y = random.randint(-100, 100)
                cand = (Fraction(x), Fraction(y))
                if try_add_point(best_points, cand):
                    new_set = best_points + [cand]
                    if check_validity_frac(new_set):
                        best_points = new_set
                        best_size = 7
                        found7 = True
                        break
    
    # Convert to required format
    result_points = []
    for (x, y) in best_points:
        # x and y are Fractions
        result_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
    
    return {'D': 1, 'points': result_points}
