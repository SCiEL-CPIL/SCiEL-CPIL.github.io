# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Verification function using exact rational arithmetic
    def is_valid(pts):
        n = len(pts)
        # Check distinct
        for i in range(n):
            for j in range(i+1, n):
                if pts[i] == pts[j]:
                    return False

        # Check rational distances (squared distances must be perfect squares of rationals)
        for i in range(n):
            for j in range(i+1, n):
                dx = Fraction(pts[i][0]) - Fraction(pts[j][0])
                dy = Fraction(pts[i][1]) - Fraction(pts[j][1])
                dist_sq = dx*dx + dy*dy
                if dist_sq < 0:
                    return False
                # Check if dist_sq is a perfect square of a rational
                # dist_sq = (a/b)^2 means dist_sq * b^2 = a^2
                # More simply: if dist_sq is a rational, check if it's a perfect square
                num = dist_sq.numerator
                den = dist_sq.denominator
                # Need num/den to be a perfect square
                # num must be a perfect square and den must be a perfect square
                # Actually, we need sqrt(num/den) to be rational
                # This happens when num/den in lowest terms has num and den both perfect squares
                # But dist_sq is already in lowest terms from Fraction
                # So we need num to be a perfect square and den to be a perfect square
                root_num = math.isqrt(num)
                if root_num * root_num != num:
                    return False
                root_den = math.isqrt(den)
                if root_den * root_den != den:
                    return False

        # Check no 3 collinear
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    x1, y1 = p1
                    x2, y2 = p2
                    x3, y3 = p3
                    if (y2 - y1) * (x3 - x1) == (y3 - y1) * (x2 - x1):
                        return False

        # Check no 4 concyclic
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        p1, p2, p3, p4 = pts[i], pts[j], pts[k], pts[l]
                        x1, y1 = p1
                        x2, y2 = p2
                        x3, y3 = p3
                        x4, y4 = p4

                        # Circle through p1, p2, p3: x^2 + y^2 + Ax + By + C = 0
                        # (x2^2+y2^2) + A*x2 + B*y2 + C = (x1^2+y1^2) + A*x1 + B*y1 + C
                        # A*(x2-x1) + B*(y2-y1) = (x1^2+y1^2) - (x2^2+y2^2)
                        # A*(x3-x1) + B*(y3-y1) = (x1^2+y1^2) - (x3^2+y3^2)

                        a1 = x2 - x1
                        b1 = y2 - y1
                        c1 = (x1*x1 + y1*y1) - (x2*x2 + y2*y2)

                        a2 = x3 - x1
                        b2 = y3 - y1
                        c2 = (x1*x1 + y1*y1) - (x3*x3 + y3*y3)

                        det = a1*b2 - a2*b1
                        if det == 0:
                            # Collinear, skip (shouldn't happen if no 3 collinear)
                            continue

                        A = (c1*b2 - c2*b1) / det
                        B = (a1*c2 - a2*c1) / det
                        C = -(x1*x1 + y1*y1) - A*x1 - B*y1

                        # Check if p4 lies on this circle
                        val = x4*x4 + y4*y4 + A*x4 + B*y4 + C
                        if val == 0:
                            return False

        return True

    # Known valid 5-point sets
    base_points_list = [
        [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)],
        [(0, 0), (56, 33), (16, 63), (0, 33), (-16, 63)],
        [(0, 0), (-3, 4), (0, 5), (3, 4), (-4, 3)],
        [(0, 0), (4, 3), (0, 5), (-4, 3), (3, 4)],
    ]

    # Verify base sets
    valid_base_sets = []
    for bp in base_points_list:
        if is_valid(bp):
            valid_base_sets.append(bp)

    if not valid_base_sets:
        # Fallback: return a simple valid 3-point set
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}

    best_points = valid_base_sets[0]
    best_score = len(valid_base_sets[0])

    # For each valid base set, try to find additional points
    for base_points in valid_base_sets:
        if time.time() > deadline - 5:
            break

        base_set = set(base_points)
        found_new_point = False

        # Strategy 1: Use rational parametrization of circles
        # For a point (x0, y0) and radius r, points on the circle can be parameterized
        # as (x0 + r*(1-t^2)/(1+t^2), y0 + r*(2t)/(1+t^2)) for rational t
        # But we need distances to ALL existing points to be rational

        # Strategy 2: Spiral search with wider range
        max_radius = 100

        while not found_new_point and time.time() < deadline - 5:
            # Generate candidates in spiral order
            for r in range(0, max_radius + 1):
                if time.time() > deadline - 5:
                    break

                candidates = []
                if r == 0:
                    candidates.append((0, 0))
                else:
                    # Points on the square boundary at radius r
                    for x in range(-r, r + 1):
                        if abs(x) == r:
                            for y in range(-r, r + 1):
                                if abs(y) == r:
                                    candidates.append((x, y))
                    # Also add some interior points for better coverage
                    for x in range(-r, r + 1):
                        for y in range(-r, r + 1):
                            if abs(x) < r and abs(y) < r:
                                # Only add a fraction to keep it manageable
                                if (x + y) % 3 == 0:
                                    candidates.append((x, y))

                # Deduplicate
                seen = set()
                unique_candidates = []
                for c in candidates:
                    if c not in seen:
                        seen.add(c)
                        unique_candidates.append(c)

                for (x, y) in unique_candidates:
                    if time.time() > deadline - 5:
                        break

                    # Skip if this point is already in base_points
                    if (x, y) in base_set:
                        continue

                    # Quick check: distances to all base points must be rational (perfect squares)
                    valid_candidate = True
                    for (xi, yi) in base_points:
                        dx = x - xi
                        dy = y - yi
                        dist_sq = dx*dx + dy*dy
                        if dist_sq < 0:
                            valid_candidate = False
                            break
                        root = math.isqrt(dist_sq)
                        if root * root != dist_sq:
                            valid_candidate = False
                            break

                    if not valid_candidate:
                        continue

                    # Check collinearity with existing points
                    collinear_found = False
                    for i in range(len(base_points)):
                        for j in range(i+1, len(base_points)):
                            p1 = (x, y)
                            p2 = base_points[i]
                            p3 = base_points[j]
                            if (p2[1]-p1[1])*(p3[0]-p1[0]) == (p3[1]-p1[1])*(p2[0]-p1[0]):
                                collinear_found = True
                                break
                        if collinear_found:
                            break

                    if collinear_found:
                        continue

                    # Check concyclicity: no 4 points (including new) should be concyclic
                    concyclic_found = False
                    for i in range(len(base_points)):
                        for j in range(i+1, len(base_points)):
                            for k in range(j+1, len(base_points)):
                                p1 = base_points[i]
                                p2 = base_points[j]
                                p3 = base_points[k]
                                p4 = (x, y)

                                x1, y1 = p1
                                x2, y2 = p2
                                x3, y3 = p3
                                x4, y4 = p4

                                a1 = x2 - x1
                                b1 = y2 - y1
                                c1 = (x1*x1 + y1*y1) - (x2*x2 + y2*y2)

                                a2 = x3 - x1
                                b2 = y3 - y1
                                c2 = (x1*x1 + y1*y1) - (x3*x3 + y3*y3)

                                det = a1*b2 - a2*b1
                                if det == 0:
                                    continue

                                A_num = c1*b2 - c2*b1
                                B_num = a1*c2 - a2*c1
                                C_num = -(x1*x1 + y1*y1)*det - A_num*x1 - B_num*y1

                                check = (x4*x4 + y4*y4)*det + A_num*x4 + B_num*y4 + C_num
                                if check == 0:
                                    concyclic_found = True
                                    break
                            if concyclic_found:
                                break
                        if concyclic_found:
                            break

                    if concyclic_found:
                        continue

                    # All checks passed! We found a valid new point
                    candidate_set = base_points + [(x, y)]
                    if is_valid(candidate_set):
                        if len(candidate_set) > best_score:
                            best_points = candidate_set
                            best_score = len(candidate_set)
                        found_new_point = True
                        break

                if found_new_point:
                    break

            if found_new_point:
                break

            max_radius += 25

    # Final verification
    if is_valid(best_points):
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best_points]}

    # Fallback
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in valid_base_sets[0]]}
# EVOLVE-BLOCK-END
