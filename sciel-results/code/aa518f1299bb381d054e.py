"""Combined program: exact rational arithmetic + fast integer grid search
for integral point sets in general position."""
import time
import random
from math import isqrt, gcd
from fractions import Fraction
from itertools import combinations

T0 = time.time()
TIME_LIMIT = 145  # safety margin before 150s

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _sqdist_int(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx * dx + dy * dy

def _collinear3_int(A, B, C):
    return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

def _conic_det(A, B, C, D):
    (x1, y1) = A
    (x2, y2) = B
    (x3, y3) = C
    (x4, y4) = D
    s1 = x1*x1 + y1*y1
    s2 = x2*x2 + y2*y2
    s3 = x3*x3 + y3*y3
    s4 = x4*x4 + y4*y4
    d = (
        x1 * y2 * (s3 - s4) - x1 * y3 * (s2 - s4) + x1 * y4 * (s2 - s3)
        - y1 * x2 * (s3 - s4) + y1 * x3 * (s2 - s4) - y1 * x4 * (s2 - s3)
        + s2 * (y3 * x4 - y4 * x3)
        - s3 * (y2 * x4 - y4 * x2)
        + s4 * (y2 * x3 - y3 * x2)
    )
    return d

def _valid_int_set(pts):
    n = len(pts)
    if n < 3:
        return False
    seen = set(pts)
    if len(seen) != n:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if not _is_square(_sqdist_int(pts[i], pts[j])):
                return False
    for a, b, c in combinations(pts, 3):
        if _collinear3_int(a, b, c) == 0:
            return False
    for a, b, c, d in combinations(pts, 4):
        if _conic_det(a, b, c, d) == 0:
            return False
    return True

def _check_rational_distances_frac(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = points[i]
            x2, y2 = points[j]
            dx = x1 - x2
            dy = y1 - y2
            sd = dx * dx + D * dy * dy
            if sd < 0:
                return False
            num, den = sd.numerator, sd.denominator
            if not _is_square(num) or not _is_square(den):
                return False
    return True

def _check_no_three_collinear_frac(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                cross = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                if cross == 0:
                    return False
    return True

def _check_no_four_concyclic_frac(points, D):
    n = len(points)
    for quad in combinations(range(n), 4):
        i, j, k, l = quad
        x1, y1 = points[i]
        x2, y2 = points[j]
        x3, y3 = points[k]
        x4, y4 = points[l]
        a11 = x2 - x1
        a12 = y2 - y1
        a21 = x3 - x1
        a22 = y3 - y1
        b1 = -((x2*x2 - x1*x1) + D*(y2*y2 - y1*y1))
        b2 = -((x3*x3 - x1*x1) + D*(y3*y3 - y1*y1))
        det = a11 * a22 - a12 * a21
        if det == 0:
            continue
        A = (b1 * a22 - b2 * a12) / det
        B = (a11 * b2 - a21 * b1) / det
        C = -(x1*x1 + D*y1*y1 + A*x1 + B*y1)
        val = x4*x4 + D*y4*y4 + A*x4 + B*y4 + C
        if val == 0:
            return False
    return True

def _valid_frac(points, D):
    if not _check_rational_distances_frac(points, D):
        return False
    if not _check_no_three_collinear_frac(points):
        return False
    if not _check_no_four_concyclic_frac(points, D):
        return False
    return True

def _format_int(pts):
    return {"D": 1, "points": [[[x, 1], [y, 1]] for (x, y) in pts]}

def _format_frac(points, D):
    result_points = []
    for x, y in points:
        result_points.append([
            [x.numerator, x.denominator],
            [y.numerator, y.denominator]
        ])
    return {'D': D, 'points': result_points}

def _search_grid_extend(base, deadline, max_radius=300):
    """Try to extend base by scanning a grid for valid new points."""
    best = base[:]
    existing = set(best)
    xs = [p[0] for p in best]
    ys = [p[1] for p in best]
    xmin = min(xs) - max_radius
    xmax = max(xs) + max_radius
    ymin = min(ys) - max_radius
    ymax = max(ys) + max_radius

    while len(best) < 12:
        if time.time() > deadline:
            break
        n = len(best)
        cand = None
        found = False
        for x in range(xmin, xmax + 1):
            if time.time() > deadline:
                break
            for y in range(ymin, ymax + 1):
                if (x, y) in existing:
                    continue
                # Check integer squared distances to all existing
                ok = True
                for p in best:
                    if not _is_square(_sqdist_int((x, y), p)):
                        ok = False
                        break
                if not ok:
                    continue
                # Check collinearity
                bad = False
                for a in range(n):
                    for b in range(a + 1, n):
                        if _collinear3_int(best[a], best[b], (x, y)) == 0:
                            bad = True
                            break
                    if bad:
                        break
                if bad:
                    continue
                # Check concyclicity with all triples
                for a, b, c in combinations(range(n), 3):
                    if _conic_det(best[a], best[b], best[c], (x, y)) == 0:
                        bad = True
                        break
                if bad:
                    continue
                cand = (x, y)
                found = True
                break
            if found:
                break
        if not found:
            # Try expanding radius
            max_radius = int(max_radius * 1.5) + 50
            xmin = min(xs) - max_radius
            xmax = max(xs) + max_radius
            ymin = min(ys) - max_radius
            ymax = max(ys) + max_radius
            if max_radius > 5000:
                break
            continue
        new_set = best + [cand]
        if _valid_int_set(new_set):
            best = new_set
            existing.add(cand)
            xs = [p[0] for p in best]
            ys = [p[1] for p in best]
            xmin = min(xs) - max_radius
            xmax = max(xs) + max_radius
            ymin = min(ys) - max_radius
            ymax = max(ys) + max_radius
        else:
            # Try a few more candidates
            count = 0
            for x in range(xmin, xmax + 1):
                if time.time() > deadline:
                    break
                if count > 200:
                    break
                for y in range(ymin, ymax + 1):
                    if (x, y) in existing or (x, y) == cand:
                        continue
                    ok = True
                    for p in best:
                        if not _is_square(_sqdist_int((x, y), p)):
                            ok = False
                            break
                    if not ok:
                        continue
                    bad = False
                    for a in range(len(best)):
                        for b in range(a + 1, len(best)):
                            if _collinear3_int(best[a], best[b], (x, y)) == 0:
                                bad = True
                                break
                        if bad:
                            break
                    if bad:
                        continue
                    for a, b, c in combinations(range(len(best)), 3):
                        if _conic_det(best[a], best[b], best[c], (x, y)) == 0:
                            bad = True
                            break
                    if bad:
                        continue
                    new_set2 = best + [(x, y)]
                    if _valid_int_set(new_set2):
                        best = new_set2
                        existing.add((x, y))
                        count = 999
                        break
                    count += 1
                if count >= 999:
                    break
            break

    return best

def _search_general_D(deadline):
    """Search for configurations with D > 1 using rational points."""
    best = None
    best_n = 0
    best_D = 1

    D_list = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 67, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 93, 94, 95, 97]

    for D in D_list:
        if time.time() > deadline:
            break
        # Try small integer x, y with D
        for x1 in range(-20, 21):
            if time.time() > deadline:
                break
            for y1 in range(-20, 21):
                if time.time() > deadline:
                    break
                p1 = (Fraction(x1), Fraction(y1))
                for x2 in range(-20, 21):
                    if time.time() > deadline:
                        break
                    for y2 in range(-20, 21):
                        if time.time() > deadline:
                            break
                        if x2 == x1 and y2 == y1:
                            continue
                        p2 = (Fraction(x2), Fraction(y2))
                        # Check distance p1-p2
                        dx = p1[0] - p2[0]
                        dy = p1[1] - p2[1]
                        sd = dx * dx + D * dy * dy
                        if sd < 0:
                            continue
                        num, den = sd.numerator, sd.denominator
                        if not _is_square(num) or not _is_square(den):
                            continue
                        # Try to add more points
                        pts = [p1, p2]
                        for x3 in range(-20, 21):
                            if time.time() > deadline:
                                break
                            for y3 in range(-20, 21):
                                if time.time() > deadline:
                                    break
                                if (x3, y3) == (x1, y1) or (x3, y3) == (x2, y2):
                                    continue
                                p3 = (Fraction(x3), Fraction(y3))
                                # Check distances to p1, p2
                                ok = True
                                for pp in pts:
                                    dx = pp[0] - p3[0]
                                    dy = pp[1] - p3[1]
                                    sd2 = dx * dx + D * dy * dy
                                    if sd2 < 0:
                                        ok = False
                                        break
                                    n2, d2 = sd2.numerator, sd2.denominator
                                    if not _is_square(n2) or not _is_square(d2):
                                        ok = False
                                        break
                                if not ok:
                                    continue
                                # Check collinearity
                                cross = (p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1])
                                if cross == 0:
                                    continue
                                pts3 = pts + [p3]
                                if len(pts3) > best_n:
                                    if _valid_frac(pts3, D):
                                        best = pts3
                                        best_n = len(pts3)
                                        best_D = D
                                # Try 4th point
                                for x4 in range(-20, 21):
                                    if time.time() > deadline:
                                        break
                                    for y4 in range(-20, 21):
                                        if time.time() > deadline:
                                            break
                                        if (x4, y4) in [(x1,y1),(x2,y2),(x3,y3)]:
                                            continue
                                        p4 = (Fraction(x4), Fraction(y4))
                                        ok = True
                                        for pp in pts3:
                                            dx = pp[0] - p4[0]
                                            dy = pp[1] - p4[1]
                                            sd3 = dx * dx + D * dy * dy
                                            if sd3 < 0:
                                                ok = False
                                                break
                                            n3, d3 = sd3.numerator, sd3.denominator
                                            if not _is_square(n3) or not _is_square(d3):
                                                ok = False
                                                break
                                        if not ok:
                                            continue
                                        # Collinearity
                                        bad = False
                                        for a in range(3):
                                            for b in range(a+1, 3):
                                                if _collinear3_frac(pts3[a], pts3[b], p4) == 0:
                                                    bad = True
                                                    break
                                            if bad:
                                                break
                                        if bad:
                                            continue
                                        # Concyclicity
                                        if _conic_frac(pts3[0], pts3[1], pts3[2], p4, D) == 0:
                                            continue
                                        pts4 = pts3 + [p4]
                                        if len(pts4) > best_n:
                                            if _valid_frac(pts4, D):
                                                best = pts4
                                                best_n = len(pts4)
                                                best_D = D
    return best, best_n, best_D

def _collinear3_frac(A, B, C):
    return (B[0]-A[0])*(C[1]-A[1]) - (B[1]-A[1])*(C[0]-A[0])

def _conic_frac(A, B, C, Dpt, D):
    x1, y1 = A
    x2, y2 = B
    x3, y3 = C
    x4, y4 = Dpt
    a11 = x2 - x1
    a12 = y2 - y1
    a21 = x3 - x1
    a22 = y3 - y1
    b1 = -((x2*x2 - x1*x1) + D*(y2*y2 - y1*y1))
    b2 = -((x3*x3 - x1*x1) + D*(y3*y3 - y1*y1))
    det = a11 * a22 - a12 * a21
    if det == 0:
        return 1
    Acoef = (b1 * a22 - b2 * a12) / det
    Bcoef = (a11 * b2 - a21 * b1) / det
    Ccoef = -(x1*x1 + D*y1*y1 + Acoef*x1 + Bcoef*y1)
    return x4*x4 + D*y4*y4 + Acoef*x4 + Bcoef*y4 + Ccoef

def construct():
    global T0
    T0 = time.time()
    deadline = T0 + TIME_LIMIT

    # Seed 5-point set (D=1)
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

    best_int = base[:]
    if not _valid_int_set(base):
        best_int = base[:3]

    # Try to extend with grid search
    if time.time() < deadline - 20:
        extended = _search_grid_extend(best_int, deadline - 20, max_radius=400)
        if len(extended) > len(best_int):
            best_int = extended

    # Format integer result
    best_result = _format_int(best_int)
    best_n = len(best_int)

    # Try general D if time remains
    if time.time() < deadline - 10:
        pts_gen, n_gen, D_gen = _search_general_D(deadline - 5)
        if pts_gen and n_gen > best_n:
            best_result = _format_frac(pts_gen, D_gen)
            best_n = n_gen

    return best_result
