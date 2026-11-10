"""Construct integral point sets (no three collinear, no four concyclic)."""
from fractions import Fraction
import random
import time
import math
from itertools import combinations

_START = time.time()
_DEADLINE = 135.0  # safety margin vs 150s limit

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _fmt(num, den):
    if den <= 0:
        num, den = -num, -den
    if den == 1:
        return [int(num), 1]
    return [int(num), int(den)]

def _dist_sq_D(a, b, D):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + D * dy * dy

def _is_perfect_square_frac(f):
    if f < 0:
        return False
    n, d = f.numerator, f.denominator
    if n == 0:
        return True
    sn = math.isqrt(n)
    sd = math.isqrt(d)
    return sn * sn == n and sd * sd == d

def _distance_rational(a, b, D):
    dsq = _dist_sq_D(a, b, D)
    return _is_perfect_square_frac(dsq)

def _collinear(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]) == 0

def _concyclic4(p, q, r, s, D):
    x1, y1 = p[0], p[1]
    x2, y2 = q[0], q[1]
    x3, y3 = r[0], r[1]
    x4, y4 = s[0], s[1]
    
    w1 = x1 * x1 + D * y1 * y1
    w2 = x2 * x2 + D * y2 * y2
    w3 = x3 * x3 + D * y3 * y3
    w4 = x4 * x4 + D * y4 * y4
    
    d123 = (x2 - x1) * (w3 - w1) - (x3 - x1) * (w2 - w1)
    if d123 != 0:
        return False
    d124 = (x2 - x1) * (w4 - w1) - (x4 - x1) * (w2 - w1)
    return d124 == 0

def _valid_add(pt, pts, D):
    for b in pts:
        if not _distance_rational(pt, b, D):
            return False
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if _collinear(pt, pts[i], pts[j]):
                return False
            for k in range(j + 1, len(pts)):
                if _concyclic4(pt, pts[i], pts[j], pts[k], D):
                    return False
    return True

def _generate_circle_points(cx, cy, r, max_coord=200):
    """Generate integer points on circle (x-cx)^2 + (y-cy)^2 = r^2"""
    points = set()
    r2 = r * r
    for dx in range(-r, r + 1):
        rem = r2 - dx * dx
        if rem < 0:
            continue
        dy = math.isqrt(rem)
        if dy * dy != rem:
            continue
        for sdy in (dy, -dy) if dy != 0 else (0,):
            x = cx + dx
            y = cy + sdy
            if abs(x) <= max_coord and abs(y) <= max_coord:
                points.add((Fraction(x), Fraction(y)))
    return points

def _generate_candidates_from_set(pts, D=1, max_r=15, max_coord=200):
    """Generate candidate points at integer distances from all existing points."""
    candidates = set()
    n = len(pts)
    if n == 0:
        return candidates
    
    # For each pair of existing points, find intersection of distance circles
    # Or: for each existing point, generate points at various integer distances
    # Then filter to those at integer distance from ALL points
    
    # Strategy: generate points at integer distance from first point,
    # then check if they're at integer distance from all others
    base = pts[0]
    for r in range(1, max_r + 1):
        circle_pts = _generate_circle_points(base[0].numerator, base[1].numerator if base[1].denominator == 1 else 0, r, max_coord)
        # Actually need to handle the Fraction properly
        cx = base[0].numerator if base[0].denominator == 1 else None
        cy = base[1].numerator if base[1].denominator == 1 else None
        if cx is None or cy is None:
            continue
        circle_pts = _generate_circle_points(cx, cy, r, max_coord)
        for pt in circle_pts:
            if any(pt == p for p in pts):
                continue
            # Check if at integer distance from all other points
            ok = True
            for other in pts[1:]:
                if not _distance_rational(pt, other, D):
                    ok = False
                    break
            if ok:
                candidates.add(pt)
    
    return candidates

_KNOWN_SETS = [
    [(0,0), (-56,33), (-16,63), (0,33), (16,63)],
    [(0,0), (5,0), (0,5), (3,4), (4,3)],
    [(0,0), (13,0), (0,13), (13,13), (8,13), (0,8)],
    [(0,0), (13,0), (0,13), (13,13), (8,13), (0,8), (13,8)],
    [(0,0), (-16,63), (0,33), (16,63), (-56,33)],
    [(0,0), (5,0), (0,5), (3,4)],
    [(0,0), (15,0), (0,15), (15,15), (9,15), (0,9)],
    [(0,0), (7,0), (0,7), (5,5), (3,4), (4,3)],
    [(0,0), (9,0), (0,9), (6,6), (3,4), (4,3), (5,5)],
    [(0,0), (11,0), (0,11), (8,8), (3,4), (4,3), (5,5), (6,6)],
    # More diverse starting sets
    [(0,0), (1,0), (0,1)],
    [(0,0), (3,0), (0,4)],
    [(0,0), (4,0), (0,3)],
    [(0,0), (5,0), (0,12)],
    [(0,0), (8,0), (0,6)],
    [(0,0), (9,0), (0,12)],
    [(0,0), (12,0), (0,5)],
    [(0,0), (12,0), (0,9)],
    [(0,0), (15,0), (0,20)],
    [(0,0), (20,0), (0,15)],
    [(0,0), (7,24), (24,7)],
    [(0,0), (9,40), (40,9)],
    [(0,0), (11,60), (60,11)],
    [(0,0), (13,84), (84,13)],
    [(0,0), (20,21), (21,20)],
    [(0,0), (28,45), (45,28)],
    [(0,0), (33,56), (56,33)],
    [(0,0), (48,55), (55,48)],
    [(0,0), (65,72), (72,65)],
    [(0,0), (84,13), (13,84)],
    [(0,0), (119,120), (120,119)],
]

def _verify_set(pts, D=1):
    n = len(pts)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i+1, n):
            if not _distance_rational(pts[i], pts[j], D):
                return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                if _collinear(pts[i], pts[j], pts[k]):
                    return False
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    if _concyclic4(pts[i], pts[j], pts[k], pts[l], D):
                        return False
    return True

def _try_build(seed):
    rng = random.Random(seed)
    D = 1
    
    best_pts = None
    best_len = 0
    
    # Try multiple base sets with different strategies
    for base_set in _KNOWN_SETS:
        if _time_left() < 5:
            break
        
        pts = [(Fraction(x), Fraction(y)) for x, y in base_set]
        if not _verify_set(pts, D):
            continue
        
        # Strategy 1: Targeted candidate generation from distance circles
        max_r = 20
        max_coord = 300
        added_any = True
        while _time_left() > 8 and len(pts) < 12 and added_any:
            added_any = False
            candidates = _generate_candidates_from_set(pts, D, max_r, max_coord)
            
            # Sort candidates by how "central" they are (closer to centroid)
            if candidates and len(pts) >= 2:
                xs = [p[0].numerator / p[0].denominator for p in pts]
                ys = [p[1].numerator / p[1].denominator for p in pts]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                cand_list = list(candidates)
                cand_list.sort(key=lambda p: (p[0].numerator/p[0].denominator - cx)**2 + (p[1].numerator/p[1].denominator - cy)**2)
            else:
                cand_list = list(candidates)
            
            for cand in cand_list:
                if any(cand == p for p in pts):
                    continue
                if _valid_add(cand, pts, D):
                    pts.append(cand)
                    added_any = True
                    if len(pts) >= 12:
                        break
            
            if not added_any:
                # Expand search radius
                max_r += 10
                max_coord += 50
                if max_r > 60:
                    break
        
        # Strategy 2: Random pool expansion if still stuck
        if len(pts) < 12 and _time_left() > 8:
            N = 80
            pool = set()
            for _ in range(2000):
                x = rng.randint(-N, N)
                y = rng.randint(-N, N)
                pool.add((Fraction(x), Fraction(y)))
            
            while _time_left() > 6 and len(pts) < 12:
                added = False
                for cand in list(pool):
                    if any(cand == p for p in pts):
                        continue
                    if _valid_add(cand, pts, D):
                        pts.append(cand)
                        added = True
                        if len(pts) >= 12:
                            break
                if not added:
                    N += 30
                    for _ in range(1000):
                        x = rng.randint(-N, N)
                        y = rng.randint(-N, N)
                        pool.add((Fraction(x), Fraction(y)))
                    if N > 500:
                        break
        
        if len(pts) > best_len:
            best_len = len(pts)
            best_pts = pts[:]
            if best_len >= 12:
                return best_pts
    
    # Fallback: random 3-point start
    if best_pts is None or best_len < 3:
        for attempt in range(30):
            if _time_left() < 5:
                break
            pts = []
            for _ in range(3):
                x = rng.randint(-50, 50)
                y = rng.randint(-50, 50)
                cand = (Fraction(x), Fraction(y))
                if any(cand == p for p in pts):
                    continue
                if all(_distance_rational(cand, p, D) for p in pts):
                    if all(not _collinear(cand, pts[i], pts[j]) for i in range(len(pts)) for j in range(i+1, len(pts))):
                        pts.append(cand)
            if len(pts) == 3:
                max_r = 20
                max_coord = 300
                added_any = True
                while _time_left() > 5 and len(pts) < 12 and added_any:
                    added_any = False
                    candidates = _generate_candidates_from_set(pts, D, max_r, max_coord)
                    for cand in candidates:
                        if any(cand == p for p in pts):
                            continue
                        if _valid_add(cand, pts, D):
                            pts.append(cand)
                            added_any = True
                            if len(pts) >= 12:
                                break
                    if not added_any:
                        max_r += 15
                        max_coord += 50
                        if max_r > 60:
                            break
                
                if len(pts) > best_len:
                    best_len = len(pts)
                    best_pts = pts[:]
    
    return best_pts if best_pts else [(Fraction(0), Fraction(0)), (Fraction(5), Fraction(0)), (Fraction(0), Fraction(5))]

def _finalize(pts, D=1):
    return {'D': D, 'points': [[_fmt(p[0].numerator, p[0].denominator),
                                _fmt(p[1].numerator, p[1].denominator)] for p in pts]}

def _verify(res):
    D = res['D']
    pts_raw = res['points']
    pts = []
    for xraw, yraw in pts_raw:
        x = Fraction(xraw[0], xraw[1])
        y = Fraction(yraw[0], yraw[1])
        if abs(x.numerator) > 10**40 or x.denominator > 10**40:
            return False
        if abs(y.numerator) > 10**40 or y.denominator > 10**40:
            return False
        pts.append((x, y))
    if len(pts) < 3 or len(pts) > 12:
        return False
    if len(set(pts)) != len(pts):
        return False
    if D > 10**12 or D < 1:
        return False
    n = D
    for p in range(2, int(math.isqrt(n)) + 1):
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
    n_pts = len(pts)
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            if not _distance_rational(pts[i], pts[j], D):
                return False
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            for k in range(j + 1, n_pts):
                if _collinear(pts[i], pts[j], pts[k]):
                    return False
    for i in range(n_pts):
        for j in range(i + 1, n_pts):
            for k in range(j + 1, n_pts):
                for l in range(k + 1, n_pts):
                    if _concyclic4(pts[i], pts[j], pts[k], pts[l], D):
                        return False
    return True

def _run():
    best = None
    best_score = -1
    
    seed = 1
    while _time_left() > 15:
        try:
            pts = _try_build(seed)
        except Exception:
            seed += 1
            continue
        res = _finalize(pts, D=1)
        if _verify(res):
            sc = len(pts)
            if sc > best_score:
                best_score = sc
                best = res
                if len(pts) >= 12:
                    return best
        seed += 1
    
    if best is None:
        points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
        best = {'D': 1, 'points': [[_fmt(x, 1), _fmt(y, 1)] for x, y in points]}
    return best

def construct():
    return _run()
