import time
import math
from itertools import combinations, product
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def is_square(n):
        if n < 0:
            return False
        if isinstance(n, Fraction):
            a = n.numerator
            b = n.denominator
            ra = math.isqrt(a)
            rb = math.isqrt(b)
            return ra * ra == a and rb * rb == b
        r = math.isqrt(n)
        return r * r == n

    def dist_sq(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return dx*dx + dy*dy

    def check_no_three_collinear(points):
        n = len(points)
        for i, j, k in combinations(range(n), 3):
            p1, p2, p3 = points[i], points[j], points[k]
            # Collinear if cross product is zero
            cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])
            if cross == 0:
                return False
        return True

    def check_no_four_concyclic(points):
        n = len(points)
        for i, j, k, l in combinations(range(n), 4):
            p1, p2, p3, p4 = points[i], points[j], points[k], points[l]
            x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
            a1 = x1*x1+y1*y1; a2 = x2*x2+y2*y2
            a3 = x3*x3+y3*y3; a4 = x4*x4+y4*y4
            det = (x2-x1)*((y3-y1)*(a4-a1) - (a3-a1)*(y4-y1)) \
                - (y2-y1)*((x3-x1)*(a4-a1) - (a3-a1)*(x4-x1)) \
                + (a2-a1)*((x3-x1)*(y4-y1) - (y3-y1)*(x4-x1))
            if det == 0:
                return False
        return True

    def validate_set(points):
        n = len(points)
        if n < 2:
            return False
        # Check all pairwise distances squared are perfect squares (rational distances)
        for i in range(n):
            for j in range(i+1, n):
                d = dist_sq(points[i], points[j])
                if not is_square(d):
                    return False
        # Check no three collinear
        if not check_no_three_collinear(points):
            return False
        # Check no four concyclic
        if not check_no_four_concyclic(points):
            return False
        return True

    def can_add_point(p, existing):
        """Check if point p can be added to existing set."""
        # Check distances
        for q in existing:
            d = dist_sq(p, q)
            if not is_square(d):
                return False
        # Check collinearity with any pair
        for i, j in combinations(range(len(existing)), 2):
            cross = (existing[i][0]-p[0])*(existing[j][1]-p[1]) - (existing[j][0]-p[0])*(existing[i][1]-p[1])
            if cross == 0:
                return False
        # Check concyclicity with any triple
        if len(existing) >= 3:
            for i, j, k in combinations(range(len(existing)), 3):
                p1, p2, p3, p4 = existing[i], existing[j], existing[k], p
                x1,y1 = p1; x2,y2 = p2; x3,y3 = p3; x4,y4 = p4
                a1 = x1*x1+y1*y1; a2 = x2*x2+y2*y2
                a3 = x3*x3+y3*y3; a4 = x4*x4+y4*y4
                det = (x2-x1)*((y3-y1)*(a4-a1) - (a3-a1)*(y4-y1)) \
                    - (y2-y1)*((x3-x1)*(a4-a1) - (a3-a1)*(x4-x1)) \
                    + (a2-a1)*((x3-x1)*(y4-y1) - (y3-y1)*(x4-x1))
                if det == 0:
                    return False
        return True

    # Start with a known valid 5-point set
    # Points: (0,0), (-56,33), (-16,63), (0,33), (16,63)
    # All distances are integers, no three collinear, no four concyclic
    seed = [(Fraction(0), Fraction(0)),
            (Fraction(-56), Fraction(33)),
            (Fraction(-16), Fraction(63)),
            (Fraction(0), Fraction(33)),
            (Fraction(16), Fraction(63))]

    if not validate_set(seed):
        # Fallback minimal set
        seed = [(Fraction(0), Fraction(0)), (Fraction(3), Fraction(4))]

    best_set = list(seed)

    def circle_through_3(p1, p2, p3):
        x1,y1 = p1; x2,y2 = p2; x3,y3 = p3
        a1 = x1*x1+y1*y1; a2 = x2*x2+y2*y2; a3 = x3*x3+y3*y3
        det = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
        if det == 0:
            return None
        D_val = ((a1-a2)*(y3-y1) - (a1-a3)*(y2-y1)) / det
        E_val = ((x2-x1)*(a1-a3) - (x3-x1)*(a1-a2)) / det
        F_val = -a1 - D_val*x1 - E_val*y1
        cx = -D_val / 2
        cy = -E_val / 2
        r2 = cx*cx + cy*cy - F_val
        return (cx, cy, r2)

    def find_point_on_circle(cx, cy, r2, base_pt, t):
        x0, y0 = base_pt
        b = y0 - t*x0
        A = 1 + t*t
        B = -2*cx + 2*t*b - 2*cy*t
        x1 = -B/A - x0
        y1 = t*x1 + b
        return (x1, y1)

    # Try circle parameterization from each triple and each base point
    triples = list(combinations(range(len(best_set)), 3))
    found_more = False
    for (i, j, k) in triples:
        if time.time() > deadline - 10:
            break
        circle = circle_through_3(best_set[i], best_set[j], best_set[k])
        if circle is None:
            continue
        cx, cy, r2 = circle
        for base_idx in range(len(best_set)):
            if time.time() > deadline - 5:
                break
            base_pt = best_set[base_idx]
            for t_num in range(-60, 61):
                if time.time() > deadline - 5:
                    break
                for t_den in range(1, 61):
                    if time.time() > deadline - 5:
                        break
                    t = Fraction(t_num, t_den)
                    p_new = find_point_on_circle(cx, cy, r2, base_pt, t)
                    is_new = True
                    for q in best_set:
                        if p_new[0] == q[0] and p_new[1] == q[1]:
                            is_new = False
                            break
                    if not is_new:
                        continue
                    if can_add_point(p_new, best_set):
                        best_set.append(p_new)
                        found_more = True
                        break
                if found_more:
                    break
            if found_more:
                break
        if found_more:
            break

    # Try again for a 7th point
    if len(best_set) >= 6:
        triples = list(combinations(range(len(best_set)), 3))
        found_more = False
        for (i, j, k) in triples:
            if time.time() > deadline - 10:
                break
            circle = circle_through_3(best_set[i], best_set[j], best_set[k])
            if circle is None:
                continue
            cx, cy, r2 = circle
            for base_idx in range(len(best_set)):
                if time.time() > deadline - 5:
                    break
                base_pt = best_set[base_idx]
                for t_num in range(-40, 41):
                    if time.time() > deadline - 5:
                        break
                    for t_den in range(1, 41):
                        if time.time() > deadline - 5:
                            break
                        t = Fraction(t_num, t_den)
                        p_new = find_point_on_circle(cx, cy, r2, base_pt, t)
                        is_new = True
                        for q in best_set:
                            if p_new[0] == q[0] and p_new[1] == q[1]:
                                is_new = False
                                break
                        if not is_new:
                            continue
                        if can_add_point(p_new, best_set):
                            best_set.append(p_new)
                            found_more = True
                            break
                    if found_more:
                        break
                if found_more:
                    break
            if found_more:
                break

    if not validate_set(best_set):
        best_set = list(seed)

    D = 1
    points_formatted = []
    for (x, y) in best_set:
        x_num = x.numerator if isinstance(x, Fraction) else x
        x_den = x.denominator if isinstance(x, Fraction) else 1
        y_num = y.numerator if isinstance(y, Fraction) else y
        y_den = y.denominator if isinstance(y, Fraction) else 1
        points_formatted.append([[x_num, x_den], [y_num, y_den]])

    return {'D': D, 'points': points_formatted}
