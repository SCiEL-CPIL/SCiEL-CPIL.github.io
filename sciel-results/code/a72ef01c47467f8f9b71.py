import time
import math
from itertools import combinations
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 140.0

    seed_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    def is_int_dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        d2 = dx*dx + dy*dy
        r = int(math.isqrt(d2))
        return r*r == d2

    def is_collinear(p1, p2, p3):
        return (p2[0]-p1[0])*(p3[1]-p1[1]) == (p2[1]-p1[1])*(p3[0]-p1[0])

    def is_concyclic(p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        a11, a12, a13 = 2*x1, 2*y1, 1
        a21, a22, a23 = 2*x2, 2*y2, 1
        a31, a32, a33 = 2*x3, 2*y3, 1
        b1 = x1*x1 + y1*y1
        b2 = x2*x2 + y2*y2
        b3 = x3*x3 + y3*y3
        
        det = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)
        if det == 0:
            return False
        
        det_A = b1*(a22*a33 - a23*a32) - a12*(b2*a33 - a23*b3) + a13*(b2*a32 - a22*b3)
        det_B = a11*(b2*a33 - a23*b3) - b1*(a21*a33 - a23*a31) + a13*(a21*b3 - b2*a31)
        det_C = a11*(a22*b3 - b2*a32) - a12*(a21*b3 - b2*a31) + b1*(a21*a32 - a22*a31)
        
        lhs = (x4*x4 + y4*y4)*det + det_A*x4 + det_B*y4 + det_C
        
        return lhs == 0

    def circle_through(p1, p2, p3):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        
        a11, a12, a13 = 2*x1, 2*y1, 1
        a21, a22, a23 = 2*x2, 2*y2, 1
        a31, a32, a33 = 2*x3, 2*y3, 1
        b1 = x1*x1 + y1*y1
        b2 = x2*x2 + y2*y2
        b3 = x3*x3 + y3*y3
        
        det = a11*(a22*a33 - a23*a32) - a12*(a21*a33 - a23*a31) + a13*(a21*a32 - a22*a31)
        if det == 0:
            return None
        
        det_A = b1*(a22*a33 - a23*a32) - a12*(b2*a33 - a23*b3) + a13*(b2*a32 - a22*b3)
        det_B = a11*(b2*a33 - a23*b3) - b1*(a21*a33 - a23*a31) + a13*(a21*b3 - b2*a31)
        det_C = a11*(a22*b3 - b2*a32) - a12*(a21*b3 - b2*a31) + b1*(a21*a32 - a22*a31)
        
        A = Fraction(det_A, det)
        B = Fraction(det_B, det)
        C = Fraction(det_C, det)
        
        return (A, B, C)

    def rational_points_on_circle(A, B, C, known_point, max_param=300):
        x0, y0 = known_point
        
        points = []
        for t_num in range(-max_param, max_param + 1):
            for t_den in range(1, max_param + 1):
                if math.gcd(abs(t_num), t_den) != 1:
                    continue
                t = Fraction(t_num, t_den)
                
                a_coef = 1 + t*t
                b_coef = -2*t*t*x0 + 2*t*y0 + A + B*t
                c_coef = t*t*x0*x0 - 2*t*x0*y0 + y0*y0 - B*t*x0 + B*y0 + C
                
                if a_coef == 0:
                    continue
                    
                x1 = -b_coef / a_coef - x0
                y1 = t * (x1 - x0) + y0
                
                if x1 != x0 or y1 != y0:
                    points.append((x1, y1))
        
        return points

    def check_point_validity(p_new, points, deadline):
        n = len(points)
        
        for p in points:
            dx = p_new[0] - p[0]
            dy = p_new[1] - p[1]
            d2 = dx*dx + dy*dy
            if d2 < 0:
                return False
            if isinstance(d2, Fraction):
                num = d2.numerator
                den = d2.denominator
                r_num = int(math.isqrt(num))
                r_den = int(math.isqrt(den))
                if r_num * r_num != num or r_den * r_den != den:
                    return False
            else:
                r = int(math.isqrt(d2))
                if r * r != d2:
                    return False
        
        for i in range(n):
            for j in range(i+1, n):
                if is_collinear(points[i], points[j], p_new):
                    return False
        
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if is_concyclic(points[i], points[j], points[k], p_new):
                        return False
        
        return True

    def find_extension_via_grid(points, max_range, deadline):
        n = len(points)
        min_x = min(p[0] for p in points) - max_range
        max_x = max(p[0] for p in points) + max_range
        min_y = min(p[1] for p in points) - max_range
        max_y = max(p[1] for p in points) + max_range

        point_set = set(points)

        for x in range(min_x, max_x + 1):
            if time.time() > deadline:
                return None
            for y in range(min_y, max_y + 1):
                p_new = (x, y)
                if p_new in point_set:
                    continue

                valid_dist = True
                for p in points:
                    if not is_int_dist(p_new, p):
                        valid_dist = False
                        break
                if not valid_dist:
                    continue

                valid_col = True
                for i in range(n):
                    for j in range(i+1, n):
                        if is_collinear(points[i], points[j], p_new):
                            valid_col = False
                            break
                    if not valid_col:
                        break
                if not valid_col:
                    continue

                valid_con = True
                for i in range(n):
                    for j in range(i+1, n):
                        for k in range(j+1, n):
                            if is_concyclic(points[i], points[j], points[k], p_new):
                                valid_con = False
                                break
                        if not valid_con:
                            break
                    if not valid_con:
                        break
                if not valid_con:
                    continue

                return p_new
        return None

    def try_add_point(points, deadline):
        n = len(points)
        
        # Strategy 1: Try circles through triples of existing points
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if time.time() > deadline:
                        return None
                    circle = circle_through(points[i], points[j], points[k])
                    if circle is None:
                        continue
                    A, B, C = circle
                    
                    candidates = rational_points_on_circle(A, B, C, points[i], max_param=100)
                    
                    for cand in candidates:
                        if abs(cand[0].numerator) > 10**8 or cand[0].denominator > 10**8:
                            continue
                        if abs(cand[1].numerator) > 10**8 or cand[1].denominator > 10**8:
                            continue
                        
                        if check_point_validity(cand, points, deadline):
                            return cand
        
        # Strategy 2: Grid search for integer points
        p_new = find_extension_via_grid(points, 500, deadline)
        if p_new:
            return p_new
        
        return None

    best_points = seed_points

    p6 = try_add_point(best_points, deadline)
    if p6:
        best_points = best_points + [p6]
        p7 = try_add_point(best_points, deadline)
        if p7:
            best_points = best_points + [p7]
            p8 = try_add_point(best_points, deadline)
            if p8:
                best_points = best_points + [p8]

    final_points = []
    for p in best_points:
        x, y = p
        if isinstance(x, Fraction):
            x = x.numerator, x.denominator
        else:
            x = x, 1
        if isinstance(y, Fraction):
            y = y.numerator, y.denominator
        else:
            y = y, 1
        final_points.append([x, y])

    return {'D': 1, 'points': final_points}
