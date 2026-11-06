import math
import time
from fractions import Fraction
from itertools import combinations

def construct():
    t0 = time.time()
    deadline = 140.0  # safety margin before 150s kill

    def time_left():
        return deadline - (time.time() - t0)

    # Known valid 5-point integral set
    base_points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    def is_square(n):
        if n < 0:
            return False, 0
        r = math.isqrt(n)
        if r * r == n:
            return True, r
        return False, 0

    def check_collinear(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    x1, y1 = pts[i]
                    x2, y2 = pts[j]
                    x3, y3 = pts[k]
                    if (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1) == 0:
                        return True
        return False

    def check_concyclic(pts):
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        a, b = pts[i]
                        c, d = pts[j]
                        e, f = pts[k]
                        g, h = pts[l]
                        # Check if four points are concyclic using determinant
                        # | x^2+y^2  x  y  1 |
                        m = [
                            [a * a + b * b, a, b, 1],
                            [c * c + d * d, c, d, 1],
                            [e * e + f * f, e, f, 1],
                            [g * g + h * h, g, h, 1]
                        ]
                        # 4x4 determinant
                        def det4(m):
                            return (m[0][0] * (m[1][1] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) - m[1][2] * (m[2][1] * m[3][3] - m[2][3] * m[3][1]) + m[1][3] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]))
                                   - m[0][1] * (m[1][0] * (m[2][2] * m[3][3] - m[2][3] * m[3][2]) - m[1][2] * (m[2][0] * m[3][3] - m[2][3] * m[3][0]) + m[1][3] * (m[2][0] * m[3][2] - m[2][2] * m[3][0]))
                                   + m[0][2] * (m[1][0] * (m[2][1] * m[3][3] - m[2][3] * m[3][1]) - m[1][1] * (m[2][0] * m[3][3] - m[2][3] * m[3][0]) + m[1][3] * (m[2][0] * m[3][1] - m[2][1] * m[3][0]))
                                   - m[0][3] * (m[1][0] * (m[2][1] * m[3][2] - m[2][2] * m[3][1]) - m[1][1] * (m[2][0] * m[3][2] - m[2][2] * m[3][0]) + m[1][2] * (m[2][0] * m[3][1] - m[2][1] * m[3][0])))
                        if det4(m) == 0:
                            return True
        return False

    def validate_set(pts):
        # Check all pairwise distances are integers
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                dx = pts[i][0] - pts[j][0]
                dy = pts[i][1] - pts[j][1]
                d2 = dx * dx + dy * dy
                ok, _ = is_square(d2)
                if not ok:
                    return False
        # Check no three collinear
        if check_collinear(pts):
            return False
        # Check no four concyclic
        if check_concyclic(pts):
            return False
        return True

    # Verify base set
    if not validate_set(base_points):
        # Fallback: just return a simple valid 3-point set
        return {'D': 1, 'points': [[[0, 1], [0, 1]], [[1, 1], [0, 1]], [[0, 1], [1, 1]]]}

    best = list(base_points)

    # Strategy 1: Brute force search for 6th point in a reasonable box
    # The base points span x: -56..16, y: 0..63
    # Search in a larger box
    search_range = 150

    found_6 = False
    for x in range(-search_range, search_range + 1):
        if time_left() < 10:
            break
        for y in range(-search_range, search_range + 1):
            new_pt = (x, y)
            if new_pt in base_points:
                continue
            # Quick distance check
            valid = True
            for p in base_points:
                dx = x - p[0]
                dy = y - p[1]
                d2 = dx * dx + dy * dy
                ok, _ = is_square(d2)
                if not ok:
                    valid = False
                    break
            if not valid:
                continue
            # Full validation
            candidate = base_points + [new_pt]
            if validate_set(candidate):
                best = candidate
                found_6 = True
                break
        if found_6 or time_left() < 10:
            break

    if len(best) == 6:
        # Try to find a 7th point
        search_range2 = 200
        found_7 = False
        for x in range(-search_range2, search_range2 + 1):
            if time_left() < 5:
                break
            for y in range(-search_range2, search_range2 + 1):
                new_pt = (x, y)
                if new_pt in best:
                    continue
                valid = True
                for p in best:
                    dx = x - p[0]
                    dy = y - p[1]
                    d2 = dx * dx + dy * dy
                    ok, _ = is_square(d2)
                    if not ok:
                        valid = False
                        break
                if not valid:
                    continue
                candidate = best + [new_pt]
                if validate_set(candidate):
                    best = candidate
                    found_7 = True
                    break
            if found_7 or time_left() < 5:
                break

    # Strategy 2: Circle-based parameterization to find additional points
    # For any three existing points, find the circle and parameterize rational points on it
    if time_left() > 15:
        # Get all circles through triples of existing points
        n = len(best)
        circles = []
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b = best[i]
                    c, d = best[j]
                    e, f = best[k]
                    # Circle: x^2 + y^2 + Ax + By + C = 0
                    # Solve for A, B, C
                    # a^2+b^2 + Aa + Bb + C = 0
                    # c^2+d^2 + Ac + Bd + C = 0
                    # e^2+f^2 + Ae + Bf + C = 0
                    # Subtract first from second:
                    # (c^2+d^2-a^2-b^2) + A(c-a) + B(d-b) = 0
                    # Subtract first from third:
                    # (e^2+f^2-a^2-b^2) + A(e-a) + B(f-b) = 0
                    A1 = c - a
                    B1 = d - b
                    C1 = a * a + b * b - c * c - d * d
                    A2 = e - a
                    B2 = f - b
                    C2 = a * a + b * b - e * e - f * f
                    det = A1 * B2 - A2 * B1
                    if det == 0:
                        continue
                    A = Fraction(C1 * B2 - C2 * B1, det)
                    B = Fraction(C1 * A2 - C2 * A1, det)
                    C = Fraction(-(a * a + b * b) - A * a - B * b)
                    # Circle center: (-A/2, -B/2), radius^2 = (A^2+B^2)/4 - C
                    circles.append((A, B, C))

        # For each circle, try to find rational points on it that are at integer distance
        # from all existing points
        # Parameterize circle: x = x0 + (1-t^2)/(1+t^2) * r, y = y0 + 2t/(1+t^2) * r
        # where (x0, y0) is center and r is radius
        # This gives rational points for rational t

        # Actually, let's use a simpler approach: for each circle, we know three points on it.
        # We can find the circle equation and then search for other integer points on it.
        # But that's expensive. Instead, let's try a different approach.

        # Strategy 3: Try to find points that are at integer distance from all existing points
        # by solving the system of equations
        # (x-xi)^2 + (y-yi)^2 = ki^2 for i=1,...,n
        # This is a system of quadratic equations. For n>=3, this determines x,y uniquely
        # (up to finite choices of ki).

        # For the 5-point set, let's try to find a 6th point by searching over possible
        # distances to the first 3 points, then checking the remaining 2.
        if time_left() > 10 and len(best) == 5:
            # For each triple of base points, the intersection of 3 circles (one for each
            # possible distance) gives at most 2 points. We can enumerate small distances.
            # This is still expensive, so let's just do a wider brute force search.
            search_range3 = 300
            for x in range(-search_range3, search_range3 + 1):
                if time_left() < 5:
                    break
                for y in range(-search_range3, search_range3 + 1):
                    new_pt = (x, y)
                    if new_pt in best:
                        continue
                    valid = True
                    for p in best:
                        dx = x - p[0]
                        dy = y - p[1]
                        d2 = dx * dx + dy * dy
                        ok, _ = is_square(d2)
                        if not ok:
                            valid = False
                            break
                    if not valid:
                        continue
                    candidate = best + [new_pt]
                    if validate_set(candidate):
                        best = candidate
                        break
                if len(best) > 5 or time_left() < 5:
                    break

    # Return the best set found
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}
