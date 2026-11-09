import time
from fractions import Fraction
from math import isqrt, gcd
from itertools import combinations
import random

_T0 = time.time()
_DEADLINE = _T0 + 145.0

def _elapsed():
    return time.time() - _T0

def _deadline_hit():
    return _elapsed() > _DEADLINE

def _is_squarefree(n):
    if n <= 0:
        return False
    for p in range(2, isqrt(n) + 1):
        if n % (p * p) == 0:
            return False
    return True

def _rational_sqrt(n):
    if n < 0:
        return None
    r = isqrt(n)
    if r * r == n:
        return (r, 1)
    return None

def _is_rational(x):
    return x.denominator == 1

def _check_integral(points, D):
    n = len(points)
    ok = True
    dists = []
    for i in range(n):
        xi, yi = points[i]
        for j in range(i + 1, n):
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            d2 = dx * dx + D * dy * dy
            if d2 < 0:
                ok = False
                break
            num = d2.numerator
            den = d2.denominator
            g = gcd(num, den)
            num //= g
            den //= g
            rn = isqrt(num)
            rd = isqrt(den)
            if rn * rn != num or rd * rd != den:
                ok = False
                break
            dists.append((rn, rd))
    if not ok:
        return False, None
    return True, dists

def _collinear(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return (x2 - x1) * (y3 - y1) == (y2 - y1) * (x3 - x1)

def _four_concyclic(p1, p2, p3, p4, D):
    M = []
    for (x, y) in (p1, p2, p3, p4):
        M.append([x * x + D * y * y, x, y, Fraction(1)])
    n = 4
    A = [row[:] for row in M]
    for c in range(n):
        pivot = None
        for r in range(c, n):
            if A[r][c] != 0:
                pivot = r
                break
        if pivot is None:
            return True
        if pivot != c:
            A[c], A[pivot] = A[pivot], A[c]
        for r in range(c + 1, n):
            factor = A[r][c] / A[c][c]
            for cc in range(c, n):
                A[r][cc] -= factor * A[c][cc]
    det = Fraction(1)
    for i in range(n):
        det *= A[i][i]
    return det == 0

def _check_general_position(points, D):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _collinear(points[i], points[j], points[k]):
                    return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _four_concyclic(points[i], points[j], points[k], points[l], D):
                        return False
    return True

def _build_result(points, D):
    out = []
    for (x, y) in points:
        out.append([ [x.numerator, x.denominator], [y.numerator, y.denominator] ])
    return {'D': D, 'points': out}

def _score(points, D):
    return len(points)

def construct():
    global _T0, _DEADLINE
    _T0 = time.time()
    _DEADLINE = _T0 + 145.0

    best = None
    best_score = -1

    # Known 5-point set with D=1
    known_5 = [(Fraction(0), Fraction(0)), (Fraction(-56), Fraction(33)),
               (Fraction(-16), Fraction(63)), (Fraction(0), Fraction(33)),
               (Fraction(16), Fraction(63))]
    
    ok, _ = _check_integral(known_5, 1)
    if ok and _check_general_position(known_5, 1):
        best = known_5
        best_score = 5

    # Strategy: Try to extend the 5-point set to 6 or more.
    # We search for new points (x, y) with integer coordinates (for D=1)
    # that have integer distances to all existing points and break general position violations.
    
    current = list(known_5)
    D = 1
    
    # Precompute squared distances between existing points? No, we need distances to new point.
    # For a new point (x, y), dist^2 to (px, py) is (x-px)^2 + (y-py)^2.
    # This must be a perfect square.
    
    # To speed up, we can iterate x, y in a box and check.
    # Box size needs to be large enough.
    # Let's try a randomized approach with larger coordinates if greedy fails.
    
    improved = True
    max_coord = 1000
    
    while improved and not _deadline_hit():
        improved = False
        # Try to add a new point
        found = False
        # Randomize search order to avoid local minima
        xs = list(range(-max_coord, max_coord + 1))
        ys = list(range(-max_coord, max_coord + 1))
        random.seed(int(_elapsed() * 1000))
        random.shuffle(xs)
        random.shuffle(ys)
        
        # Limit number of attempts per iteration to avoid wasting time
        attempts = 0
        max_attempts = 50000
        
        for x in xs:
            if attempts > max_attempts:
                break
            if _deadline_hit():
                break
            for y in ys:
                attempts += 1
                if attempts > max_attempts:
                    break
                if _deadline_hit():
                    break
                new_pt = (Fraction(x), Fraction(y))
                if new_pt in current:
                    continue
                
                # Check integral distances
                valid = True
                for (px, py) in current:
                    dx = Fraction(x) - px
                    dy = Fraction(y) - py
                    d2 = dx * dx + dy * dy
                    num = d2.numerator
                    den = d2.denominator
                    g = gcd(num, den)
                    num //= g
                    den //= g
                    rn = isqrt(num)
                    rd = isqrt(den)
                    if rn * rn != num or rd * rd != den:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check no three collinear
                n = len(current)
                collinear_ok = True
                for i in range(n):
                    for j in range(i + 1, n):
                        if _collinear(current[i], current[j], new_pt):
                            collinear_ok = False
                            break
                    if not collinear_ok:
                        break
                if not collinear_ok:
                    continue
                
                # Check no four concyclic
                concyclic_ok = True
                for i in range(n):
                    for j in range(i + 1, n):
                        for k in range(j + 1, n):
                            if _four_concyclic(current[i], current[j], current[k], new_pt, D):
                                concyclic_ok = False
                                break
                        if not concyclic_ok:
                            break
                    if not concyclic_ok:
                        break
                if not concyclic_ok:
                    continue
                
                # Valid
                current.append(new_pt)
                improved = True
                found = True
                break
            if found:
                break
        
        if improved:
            if len(current) > best_score:
                best_score = len(current)
                best = list(current)
                # Increase box size for next iteration?
                max_coord += 100
        else:
            # If we couldn't find a point in the current box, maybe try a different D or larger box?
            # For now, break out of the loop if no improvement.
            break

    # If we found a better set, return it
    if best is not None and best_score >= 5:
        return _build_result(best, D)

    # Fallback
    return _build_result(known_5, 1)
