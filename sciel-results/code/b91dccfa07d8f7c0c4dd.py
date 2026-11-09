"""Integral point sets in general position - search for maximum size.

Strategy:
  1. Use the known 5-point set as a strong starting point.
  2. Try systematic extensions using Pythagorean-tuple candidates.
  3. If a 6th point is found, try to extend further.
  4. If no 6-point set found, return the best (5-point) set.
"""
import time
import random
from fractions import Fraction
from math import gcd, isqrt

_DEADLINE = time.time() + 125  # 125s wall-clock, 25s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _gcd(a, b):
    return gcd(abs(a), abs(b))

def _check_valid(points):
    """Check all constraints: distinct, rational distances, no 3 collinear, no 4 concyclic."""
    n = len(points)
    if n < 3:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            d2 = dx * dx + dy * dy
            if d2 <= 0:
                return False
            if not _is_square(d2):
                return False
    # No three collinear
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                x1, y1 = points[i]
                x2, y2 = points[j]
                x3, y3 = points[k]
                if (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) == 0:
                    return False
    # No four concyclic
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    x3, y3 = points[k]
                    x4, y4 = points[l]
                    a1 = x1 * x1 + y1 * y1
                    a2 = x2 * x2 + y2 * y2
                    a3 = x3 * x3 + y3 * y3
                    a4 = x4 * x4 + y4 * y4
                    b2 = a2 - a1
                    b3 = a3 - a1
                    b4 = a4 - a1
                    c2 = x2 - x1
                    c3 = x3 - x1
                    c4 = x4 - x1
                    d2v = y2 - y1
                    d3v = y3 - y1
                    d4v = y4 - y1
                    det3 = (b2 * (c3 * d4v - d3v * c4)
                            - c2 * (b3 * d4v - d3v * b4)
                            + d2v * (b3 * c4 - c3 * b4))
                    if det3 == 0:
                        return False
    return True

def _generate_pyth_candidates(center, max_radius=200):
    """Generate integer points at integer distance from center."""
    cx, cy = center
    candidates = []
    r = 1
    while r * r <= max_radius * max_radius:
        r2 = r * r
        for dx in range(-r, r + 1):
            dy2 = r2 - dx * dx
            if dy2 < 0:
                continue
            dy = isqrt(dy2)
            if dy * dy != dy2:
                continue
            for sy in ([dy, -dy] if dy > 0 else [0]):
                candidates.append((cx + dx, cy + sy))
        r += 1
    return candidates

def _try_extend(current, max_points=12, time_budget_s=110):
    """Try to add points to the set, return extended set."""
    current = list(current)
    n = len(current)
    if n < 3 or not _check_valid(current):
        return current
    
    deadline_inner = time.time() + time_budget_s
    tried_global = set()
    
    while n < max_points and time.time() < deadline_inner and time.time() < _DEADLINE - 15:
        found = False
        
        # Strategy 1: Try Pythagorean candidates from each existing point
        for px, py in current:
            cands = _generate_pyth_candidates((px, py), max_radius=300)
            random.shuffle(cands)
            for cx, cy in cands:
                if time.time() > deadline_inner or time.time() > _DEADLINE - 15:
                    break
                if (cx, cy) in current or (cx, cy) in tried_global:
                    continue
                tried_global.add((cx, cy))
                # Quick distance check
                ok = True
                for qx, qy in current:
                    dx = cx - qx
                    dy = cy - qy
                    if not _is_square(dx * dx + dy * dy):
                        ok = False
                        break
                if not ok:
                    continue
                new_set = current + [(cx, cy)]
                if _check_valid(new_set):
                    current.append((cx, cy))
                    n += 1
                    found = True
                    break
            if found:
                break
        if found:
            tried_global = set()
            continue
        
        # Strategy 2: Random search
        for _ in range(100000):
            if time.time() > deadline_inner or time.time() > _DEADLINE - 15:
                break
            # Choose center near existing points
            base_x = random.choice([p[0] for p in current])
            base_y = random.choice([p[1] for p in current])
            cx = base_x + random.randint(-150, 150)
            cy = base_y + random.randint(-150, 150)
            if (cx, cy) in current or (cx, cy) in tried_global:
                continue
            tried_global.add((cx, cy))
            ok = True
            for qx, qy in current:
                dx = cx - qx
                dy = cy - qy
                if not _is_square(dx * dx + dy * dy):
                    ok = False
                    break
            if not ok:
                continue
            new_set = current + [(cx, cy)]
            if _check_valid(new_set):
                current.append((cx, cy))
                n += 1
                found = True
                break
        
        if not found:
            break
    
    return current

def _try_hexagon_perturbation():
    """
    Try the regular hexagon idea in Q(sqrt(3)).
    A regular hexagon has 6 concyclic points (outer vertices), so it's invalid.
    But we can try: take 5 of the 6 hexagon vertices (which might avoid concyclicity issues
    if we break symmetry), or perturb one vertex.
    
    Actually, any 4 vertices of a regular hexagon on the same circle are concyclic.
    So the hexagon approach needs careful handling.
    
    Instead, let's try a different known 6-point configuration if one exists.
    """
    # One known approach: use a lattice based on Pythagorean triples
    # Try to find 6 points manually
    
    # The 5-point set:
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    if not _check_valid(base):
        return None
    
    # Try to add a 6th point using systematic search
    tried = set()
    for px, py in base:
        for r in range(1, 400):
            r2 = r * r
            for dx in range(-r, r + 1):
                dy2 = r2 - dx * dx
                if dy2 < 0:
                    continue
                dy = isqrt(dy2)
                if dy * dy != dy2:
                    continue
                for sy in ([dy, -dy] if dy > 0 else [0]):
                    cx = px + dx
                    cy = py + sy
                    key = (cx, cy)
                    if key in base or key in tried:
                        continue
                    tried.add(key)
                    # Check distances
                    ok = True
                    for qx, qy in base:
                        ddx = cx - qx
                        ddy = cy - qy
                        if not _is_square(ddx * ddx + ddy * ddy):
                            ok = False
                            break
                    if not ok:
                        continue
                    new_set = base + [(cx, cy)]
                    if _check_valid(new_set):
                        return new_set
    return None

def construct():
    global _DEADLINE
    _DEADLINE = time.time() + 125
    
    # Start with the known 5-point set
    base_5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    
    best = None
    if _check_valid(base_5):
        best = list(base_5)
    
    # Try to extend the 5-point set
    if best is not None and len(best) == 5:
        extended = _try_extend(best, max_points=12, time_budget_s=100)
        if len(extended) > len(best):
            best = extended
    
    # If still 5, try the hexagon-based perturbation
    if best is None or len(best) <= 5:
        hex_result = _try_hexagon_perturbation()
        if hex_result is not None and len(hex_result) > (len(best) if best else 0):
            best = hex_result
    
    # Try other starting configurations
    if _time_left() > 20:
        other_starts = [
            [(0, 0), (5, 0), (0, 12)],
            [(0, 0), (3, 4), (5, 0)],
            [(0, 0), (7, 24), (15, 8)],
            [(0, 0), (9, 12), (15, 0)],
            [(0, 0), (12, 5), (13, 0)],
            [(0, 0), (20, 21), (29, 0)],
            [(0, 0), (8, 15), (17, 0)],
            [(0, 0), (11, 60), (61, 0)],
            [(0, 0), (28, 45), (53, 0)],
            [(0, 0), (33, 56), (65, 0)],
            [(0, 0), (48, 55), (73, 0)],
            [(0, 0), (65, 72), (97, 0)],
            [(0, 0), (20, 99), (101, 0)],
        ]
        for start in other_starts:
            if _time_left() < 10:
                break
            if _check_valid(start):
                ext = _try_extend(start, max_points=12, time_budget_s=min(_time_left() - 10, 50))
                if best is None or len(ext) > len(best):
                    best = ext
    
    # Fallback
    if best is None or len(best) < 3:
        best = base_5
    
    if not _check_valid(best):
        best = base_5
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}
