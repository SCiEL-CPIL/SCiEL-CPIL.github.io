import time
import random
from fractions import Fraction
from itertools import combinations
import math

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s buffer
    
    # Initial 5 points
    points = [
        (Fraction(0), Fraction(0)),
        (Fraction(-56), Fraction(33)),
        (Fraction(-16), Fraction(63)),
        (Fraction(0), Fraction(33)),
        (Fraction(16), Fraction(63))
    ]
    
    best_points = points
    best_count = len(points)
    
    def is_rational_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        dist_sq = dx*dx + dy*dy
        n = dist_sq.numerator
        d = dist_sq.denominator
        if n < 0 or d < 0:
            return False
        sqrt_n = math.isqrt(n)
        sqrt_d = math.isqrt(d)
        return sqrt_n * sqrt_n == n and sqrt_d * sqrt_d == d

    def are_collinear(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        return (x3 - x1) * (y2 - y1) == (y3 - y1) * (x2 - x1)

    def are_concyclic(p1, p2, p3, p4):
        def det4(m):
            a, b, c, d = m[0]
            e, f, g, h = m[1]
            i, j, k, l = m[2]
            m_, n, o, p = m[3]
            return (a * (f * (k * p - l * o) - g * (j * p - l * n) + h * (j * o - k * n)) -
                    b * (e * (k * p - l * o) - g * (i * p - l * m_) + h * (i * o - k * m_)) +
                    c * (e * (j * p - l * n) - f * (i * p - l * m_) + h * (i * n - j * m_)) -
                    d * (e * (j * o - k * n) - f * (i * o - k * m_) + g * (i * n - j * m_)))

        m = [
            [p1[0]**2 + p1[1]**2, p1[0], p1[1], Fraction(1)],
            [p2[0]**2 + p2[1]**2, p2[0], p2[1], Fraction(1)],
            [p3[0]**2 + p3[1]**2, p3[0], p3[1], Fraction(1)],
            [p4[0]**2 + p4[1]**2, p4[0], p4[1], Fraction(1)]
        ]
        return det4(m) == 0

    def find_new_point(points):
        # Try to find a point at rational distance to all points in 'points'
        # Use 3 points to determine the candidate
        n = len(points)
        if n < 3:
            return None
            
        # Pick 3 non-collinear points
        base_pts = None
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if not are_collinear(points[i], points[j], points[k]):
                        base_pts = (points[i], points[j], points[k])
                        break
                if base_pts:
                    break
            if base_pts:
                break
                
        if not base_pts:
            return None
            
        A, B, C = base_pts
        
        # We want to find a point P such that dist(P,A), dist(P,B), dist(P,C) are rational.
        # Let dist(P,A)^2 = rA^2, etc.
        # We iterate over small integer rA, rB, rC.
        
        # To avoid searching infinite space, we bound the search.
        # The new point should be reasonably close.
        # Let's try rA, rB, rC up to 200.
        
        # Optimization: Precompute A^2, B^2, C^2
        Ax, Ay = A
        Bx, By = B
        Cx, Cy = C
        A2 = Ax*Ax + Ay*Ay
        B2 = Bx*Bx + By*By
        C2 = Cx*Cx + Cy*Cy
        
        # Equations:
        # 2(Bx-Ax)x + 2(By-Ay)y = rA^2 - rB^2 - A2 + B2
        # 2(Cx-Ax)x + 2(Cy-Ay)y = rA^2 - rC^2 - A2 + C2
        
        # Let:
        # a1 = 2(Bx-Ax), b1 = 2(By-Ay), c1 = -A2 + B2
        # a2 = 2(Cx-Ax), b2 = 2(Cy-Ay), c2 = -A2 + C2
        # a1 x + b1 y = rA^2 - rB^2 + c1
        # a2 x + b2 y = rA^2 - rC^2 + c2
        
        a1 = 2 * (Bx - Ax)
        b1 = 2 * (By - Ay)
        c1 = -A2 + B2
        a2 = 2 * (Cx - Ax)
        b2 = 2 * (Cy - Ay)
        c2 = -A2 + C2
        
        det = a1 * b2 - a2 * b1
        if det == 0:
            return None
            
        # x = ( (rA^2 - rB^2 + c1) * b2 - (rA^2 - rC^2 + c2) * b1 ) / det
        # y = ( a1 * (rA^2 - rC^2 + c2) - a2 * (rA^2 - rB^2 + c1) ) / det
        
        # We iterate rA, rB, rC
        # To keep it fast, we limit the range.
        max_r = 300
        
        for rA in range(1, max_r + 1):
            rA2 = rA * rA
            for rB in range(1, max_r + 1):
                rB2 = rB * rB
                # Quick check: rA^2 - rB^2 + c1 must be "compatible"
                # Actually, just compute x, y
                term1 = rA2 - rB2 + c1
                # If term1 is not divisible by gcd(a1, b1) in some sense? No, just compute.
                for rC in range(1, max_r + 1):
                    rC2 = rC * rC
                    term2 = rA2 - rC2 + c2
                    
                    # x_num = term1 * b2 - term2 * b1
                    # y_num = a1 * term2 - a2 * term1
                    x_num = term1 * b2 - term2 * b1
                    y_num = a1 * term2 - a2 * term1
                    
                    x = Fraction(x_num, det)
                    y = Fraction(y_num, det)
                    
                    # Check if distinct from existing
                    if any((x, y) == p for p in points):
                        continue
                        
                    # Check rational distance to ALL existing points
                    valid = True
                    for p in points:
                        if not is_rational_dist((x, y), p):
                            valid = False
                            break
                    if not valid:
                        continue
                        
                    # Check collinearity
                    for i in range(len(points)):
                        for j in range(i+1, len(points)):
                            if are_collinear((x, y), points[i], points[j]):
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                        
                    # Check concyclicity
                    for i in range(len(points)):
                        for j in range(i+1, len(points)):
                            for k in range(j+1, len(points)):
                                if are_concyclic((x, y), points[i], points[j], points[k]):
                                    valid = False
                                    break
                            if not valid:
                                break
                        if not valid:
                            break
                    if not valid:
                        continue
                        
                    return (x, y)
        return None

    # Search loop
    while time.time() < deadline and len(points) < 8:
        new_point = find_new_point(points)
        if new_point:
            points.append(new_point)
            if len(points) > best_count:
                best_points = points[:]
                best_count = len(points)
        else:
            # If we can't find a new point with the current base, 
            # maybe try a different base or random search?
            # For now, if find_new_point fails, we might be stuck.
            # Let's try a few random bases if we get stuck.
            # Or break if we've tried many times.
            # To avoid infinite loop, we can break if no progress.
            # But find_new_point is deterministic for a given set of points?
            # No, it iterates rA, rB, rC.
            # If it returns None, no point with small integer distances to the base exists.
            # We could try increasing max_r, but that's slow.
            # Let's try random search for a bit.
            
            # Simple random search fallback
            attempts = 0
            found = False
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            min_x = min(xs); max_x = max(xs)
            min_y = min(ys); max_y = max(ys)
            width = max_x - min_x
            height = max_y - min_y
            
            while attempts < 1000 and not found and time.time() < deadline:
                attempts += 1
                x_num = random.randint(int((min_x - width) * 100), int((max_x + width) * 100))
                x_den = random.randint(1, 50)
                y_num = random.randint(int((min_y - height) * 100), int((max_y + height) * 100))
                y_den = random.randint(1, 50)
                x = Fraction(x_num, x_den)
                y = Fraction(y_num, y_den)
                np = (x, y)
                if any(np == p for p in points):
                    continue
                valid = True
                for p in points:
                    if not is_rational_dist(np, p):
                        valid = False
                        break
                if valid:
                    for i in range(len(points)):
                        for j in range(i+1, len(points)):
                            if are_collinear(np, points[i], points[j]):
                                valid = False
                                break
                        if not valid: break
                if valid:
                    for i in range(len(points)):
                        for j in range(i+1, len(points)):
                            for k in range(j+1, len(points)):
                                if are_concyclic(np, points[i], points[j], points[k]):
                                    valid = False
                                    break
                            if not valid: break
                        if not valid: break
                if valid:
                    points.append(np)
                    if len(points) > best_count:
                        best_points = points[:]
                        best_count = len(points)
                    found = True
                    break
            if not found:
                # Stuck
                break

    D = 1
    out_points = []
    for p in best_points:
        x, y = p
        out_points.append([[x.numerator, x.denominator], [y.numerator, y.denominator]])
        
    return {'D': D, 'points': out_points}
