"""
Rewritten program to find integral point sets in general position.
Uses algebraic constraints to efficiently search for a 6th point extending a known 5-point set.
"""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

# Known 5-point integral set (no 3 collinear, no 4 concyclic)
_BASE5 = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]

def _is_sq(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _no_three_collinear(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                dx1 = pts[j][0] - pts[i][0]
                dy1 = pts[j][1] - pts[i][1]
                dx2 = pts[k][0] - pts[i][0]
                dy2 = pts[k][1] - pts[i][1]
                if dx1*dy2 - dy1*dx2 == 0:
                    return False
    return True

def _no_four_concyclic(pts):
    n = len(pts)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for m in range(k+1, n):
                    a = pts[i]; b = pts[j]; c = pts[k]; d = pts[m]
                    x1,y1 = a; x2,y2 = b; x3,y3 = c
                    a11 = 2*(x1-x2); a12 = 2*(y1-y2); b1 = x2*x2+y2*y2-x1*x1-y1*y1
                    a21 = 2*(x1-x3); a22 = 2*(y1-y3); b2 = x3*x3+y3*y3-x1*x1-y1*y1
                    det = a11*a22 - a12*a21
                    if det == 0:
                        continue
                    A_num = b1*a22 - b2*a12
                    B_num = a11*b2 - a21*b1
                    C_num = det*(-x1*x1-y1*y1) - A_num*x1 - B_num*y1
                    if det*(d[0]*d[0]+d[1]*d[1]) + A_num*d[0] + B_num*d[1] + C_num == 0:
                        return False
    return True

def _valid(pts):
    if not _no_three_collinear(pts):
        return False
    if not _no_four_concyclic(pts):
        return False
    return True

def _find_6th_point(base_pts, deadline):
    """
    Search for a 6th point (x,y) such that distances to all 5 base points are integers.
    Uses the constraint from points P0=(0,0) and P3=(0,33) in _BASE5.
    P0: (0,0)
    P3: (0,33)
    r0^2 = x^2 + y^2
    r3^2 = x^2 + (y-33)^2
    r3^2 - r0^2 = (y-33)^2 - y^2 = y^2 - 66y + 1089 - y^2 = 1089 - 66y
    So 66y = 1089 - (r3^2 - r0^2)
    y = (1089 - r3^2 + r0^2) / 66
    """
    # Base points for reference
    # P0: (0,0)
    # P1: (-56, 33)
    # P2: (-16, 63)
    # P3: (0, 33)
    # P4: (16, 63)
    
    # We iterate over possible integer distances r0 and r3
    # r0 = dist((x,y), (0,0))
    # r3 = dist((x,y), (0,33))
    
    max_r = 2000 # Upper bound for search
    found = None
    
    for r0 in range(1, max_r):
        if time.time() > deadline:
            break
        for r3 in range(1, max_r):
            if time.time() > deadline:
                break
            num = 1089 - r3*r3 + r0*r0
            if num % 66 != 0:
                continue
            y = num // 66
            # Check if x^2 is a square
            x2 = r0*r0 - y*y
            if x2 < 0:
                continue
            x = math.isqrt(x2)
            if x*x != x2:
                continue
            
            # We have candidate (x, y) and (-x, y)
            candidates = []
            if x != 0:
                candidates.append((x, y))
                candidates.append((-x, y))
            else:
                candidates.append((x, y))
                
            for cand in candidates:
                # Check if cand is in base
                if cand in base_pts:
                    continue
                # Check distances to other base points
                ok = True
                for i in range(len(base_pts)):
                    if i == 0 or i == 3: # Already checked via r0, r3
                        continue
                    px, py = base_pts[i]
                    dx = cand[0] - px
                    dy = cand[1] - py
                    d2 = dx*dx + dy*dy
                    if not _is_sq(d2):
                        ok = False
                        break
                if ok:
                    full_set = list(base_pts) + [cand]
                    if _valid(full_set):
                        return full_set
    return None

def construct():
    start = time.time()
    deadline = start + 145.0
    
    best = list(_BASE5)
    best_n = 5
    
    # Try to find a 6th point using algebraic constraints
    found_6 = _find_6th_point(best, deadline)
    if found_6:
        best = found_6
        best_n = 6
        
        # Try to find a 7th point using grid search on the new 6-point set
        # This is a fallback, as algebraic constraints for 7th point are complex
        R = 100
        found_7 = None
        base6 = best
        for x in range(-R, R+1):
            if time.time() > deadline:
                break
            for y in range(-R, R+1):
                if time.time() > deadline:
                    break
                if (x, y) in base6:
                    continue
                ok = True
                for i in range(6):
                    dx = x - base6[i][0]
                    dy = y - base6[i][1]
                    d2 = dx*dx + dy*dy
                    if not _is_sq(d2):
                        ok = False
                        break
                if not ok:
                    continue
                cand = list(base6) + [(x, y)]
                if _valid(cand):
                    found_7 = cand
                    break
            if found_7:
                break
        if found_7:
            best = found_7
            best_n = 7
    else:
        # If algebraic search fails, try a wider grid search for 6th point
        R = 500
        found_6 = None
        base5 = best
        for x in range(-R, R+1):
            if time.time() > deadline:
                break
            for y in range(-R, R+1):
                if time.time() > deadline:
                    break
                if (x, y) in base5:
                    continue
                ok = True
                for i in range(5):
                    dx = x - base5[i][0]
                    dy = y - base5[i][1]
                    d2 = dx*dx + dy*dy
                    if not _is_sq(d2):
                        ok = False
                        break
                if not ok:
                    continue
                cand = list(base5) + [(x, y)]
                if _valid(cand):
                    found_6 = cand
                    break
            if found_6:
                break
        if found_6:
            best = found_6
            best_n = 6

    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}
# EVOLVE-BLOCK-END
