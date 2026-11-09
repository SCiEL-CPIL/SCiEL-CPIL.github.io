"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction
from itertools import combinations

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _is_rational_square(fr):
    if fr < 0:
        return False
    if fr == 0:
        return True
    return _is_square(fr.numerator) and _is_square(fr.denominator)

def _dist2(D, p1, p2):
    x1 = Fraction(p1[0], p1[1])
    x2 = Fraction(p2[0], p2[1])
    y1 = Fraction(p1[2], p1[3])
    y2 = Fraction(p2[2], p2[3])
    dx = x1 - x2
    dy = y1 - y2
    return dx * dx + D * dy * dy

def _validate(D, points):
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if points[i] == points[j]:
                return False
            d2 = _dist2(D, points[i], points[j])
            if not _is_rational_square(d2) or d2 == 0:
                return False
    for i, j, k in combinations(range(n), 3):
        x1 = Fraction(points[i][0], points[i][1])
        y1 = Fraction(points[i][2], points[i][3])
        x2 = Fraction(points[j][0], points[j][1])
        y2 = Fraction(points[j][2], points[j][3])
        x3 = Fraction(points[k][0], points[k][1])
        y3 = Fraction(points[k][2], points[k][3])
        if (x2 - x1) * (y3 - y1) == (x3 - x1) * (y2 - y1):
            return False
    for i, j, k, l in combinations(range(n), 4):
        rows = []
        for m in [i, j, k, l]:
            x = Fraction(points[m][0], points[m][1])
            y = Fraction(points[m][2], points[m][3])
            rows.append((x * x + D * y * y, x, y))
        a, b, c = rows[0]
        d, e, f = rows[1]
        g, h, i2 = rows[2]
        j2, k2, l2 = rows[3]
        Q = ((a - j2) * ((e - k2) * (i2 - l2) - (f - l2) * (h - k2))
            - (b - k2) * ((d - j2) * (i2 - l2) - (f - l2) * (g - j2))
            + (c - l2) * ((d - j2) * (h - k2) - (e - k2) * (g - j2)))
        if Q == 0:
            return False
    return True

def _find_6_clique(D, candidates, deadline):
    """Find a 6-clique in the compatibility graph."""
    n = len(candidates)
    if n < 6:
        return None
    
    neighbors = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d2 = _dist2(D, candidates[i], candidates[j])
            if _is_rational_square(d2) and d2 > 0:
                neighbors[i].add(j)
                neighbors[j].add(i)
    
    # Order by degree
    order = sorted(range(n), key=lambda i: len(neighbors[i]), reverse=True)
    
    found = None
    
    def backtrack(clique, remaining):
        nonlocal found
        if found:
            return
        if len(clique) >= 6:
            found = clique[:]
            return
        if not remaining or time.time() > deadline:
            return
        
        for idx, v in enumerate(remaining):
            if all(u in neighbors[v] for u in clique):
                new_clique = clique + [v]
                new_remaining = remaining[idx+1:]
                backtrack(new_clique, new_remaining)
                if found:
                    return
    
    backtrack([], order)
    return found

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    # Try several D values with small candidate sets
    D_list = [2, 3, 5, 6, 7, 10, 11, 13, 14, 15, 17, 19, 21, 22, 23, 26, 29, 30, 31, 33, 34, 35, 37, 38, 39, 41, 42, 43, 46, 47, 51, 53, 55, 57, 58, 59, 61, 62, 65, 66, 69, 70, 71, 73, 74, 77, 78, 79, 82, 83, 85, 86, 87, 89, 90, 91, 93, 94, 95, 97]
    
    for D in D_list:
        if time.time() > deadline:
            break
        
        # Generate candidates
        candidates = []
        for x_num in range(-8, 9):
            for x_den in [1, 2, 3]:
                for y_num in range(-8, 9):
                    for y_den in [1, 2, 3]:
                        candidates.append((x_num, x_den, y_num, y_den))
        candidates = list(set(candidates))
        
        # Filter
        filtered = []
        for c in candidates:
            x = Fraction(c[0], c[1])
            y = Fraction(c[2], c[3])
            if abs(x) <= 6 and abs(y) <= 6:
                filtered.append(c)
        candidates = filtered[:150]
        
        if len(candidates) < 6:
            continue
        
        found = _find_6_clique(D, candidates, deadline)
        if found:
            result = [candidates[i] for i in found]
            if _validate(D, result):
                return {'D': D, 'points': [[[p[0], p[1]], [p[2], p[3]]] for p in result]}
    
    # Fallback: 5-point set
    points = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in points]}
# EVOLVE-BLOCK-END

