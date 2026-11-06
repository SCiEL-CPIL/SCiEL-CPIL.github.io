"""Combines the seed 5-point set with a systematic search for larger integral point sets.
Uses a precomputed squared-distance table to the base points for fast 6th-point search."""
# EVOLVE-BLOCK-START
import time
import math
import random
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
                    # Circle through a,b,c: x^2+y^2+Ax+By+C=0
                    # Solve for A,B,C from 3 equations
                    # (a-b): (ax-bx)^2+(ay-by)^2 + A(ax-bx) + B(ay-by) = 0
                    # Use linear system
                    x1,y1 = a; x2,y2 = b; x3,y3 = c
                    # 2A(x1-x2) + 2B(y1-y2) = x2^2+y2^2 - x1^2-y1^2
                    # 2A(x1-x3) + 2B(y1-y3) = x3^2+y3^2 - x1^2-y1^2
                    a11 = 2*(x1-x2); a12 = 2*(y1-y2); b1 = x2*x2+y2*y2-x1*x1-y1*y1
                    a21 = 2*(x1-x3); a22 = 2*(y1-y3); b2 = x3*x3+y3*y3-x1*x1-y1*y1
                    det = a11*a22 - a12*a21
                    if det == 0:
                        continue
                    # A = (b1*a22 - b2*a12)/det, B = (a11*b2 - a21*b1)/det
                    # Check if d satisfies: x^2+y^2+Ax+By+C=0 where C = -x1^2-y1^2-Ax1-By1
                    # Multiply by det to avoid fractions
                    A_num = b1*a22 - b2*a12
                    B_num = a11*b2 - a21*b1
                    # C_num = det*(-x1^2-y1^2) - A_num*x1 - B_num*y1
                    C_num = det*(-x1*x1-y1*y1) - A_num*x1 - B_num*y1
                    # Check d: det*(xd^2+yd^2) + A_num*xd + B_num*yd + C_num == 0
                    if det*(d[0]*d[0]+d[1]*d[1]) + A_num*d[0] + B_num*d[1] + C_num == 0:
                        return False
    return True

def _valid(pts):
    if not _no_three_collinear(pts):
        return False
    if not _no_four_concyclic(pts):
        return False
    return True

def construct():
    start = time.time()
    deadline = start + 145.0  # 5s safety margin
    
    # Precompute squared distances from base points to lattice points in a grid
    # For a candidate (x,y), d2_i = (x-xi)^2 + (y-yi)^2 must be a perfect square
    # Expand: x^2+y^2 - 2x*xi - 2y*yi + xi^2+yi^2 = s_i^2
    # So x^2+y^2 - 2x*xi - 2y*yi + (xi^2+yi^2) must be a square for all i
    
    # Search for a 6th point by scanning a grid
    # Use symmetry: the base set is roughly symmetric about x=0
    # Search range: try progressively larger ranges
    
    best = list(_BASE5)
    best_n = 5
    
    # Strategy 1: Grid search for 6th point with integer coordinates
    # Precompute for each base point: xi^2+yi^2
    base_sq = [xi*xi + yi*yi for xi, yi in _BASE5]
    
    # For candidate (x,y), need (x-xi)^2+(y-yi)^2 to be a square for all i
    # This is: x^2+y^2 - 2x*xi - 2y*yi + base_sq[i] = square
    
    # Scan a grid. Use the fact that distances to 5 points constrain heavily.
    # Try a range of about 300x300 centered at origin
    R = 250
    found_6 = None
    
    for x in range(-R, R+1):
        if time.time() > deadline:
            break
        for y in range(-R, R+1):
            if time.time() > deadline:
                break
            # Quick check: must not be in base
            if (x, y) in _BASE5:
                continue
            ok = True
            for i in range(5):
                dx = x - _BASE5[i][0]
                dy = y - _BASE5[i][1]
                d2 = dx*dx + dy*dy
                if not _is_sq(d2):
                    ok = False
                    break
            if not ok:
                continue
            cand = list(_BASE5) + [(x, y)]
            if _valid(cand):
                found_6 = cand
                break
        if found_6:
            break
    
    if found_6:
        best = found_6
        best_n = 6
    
    # Strategy 2: Try to find a 7th point if we have 6
    if best_n == 6:
        base6 = best
        base6_sq = [xi*xi + yi*yi for xi, yi in base6]
        R2 = 200
        found_7 = None
        for x in range(-R2, R2+1):
            if time.time() > deadline:
                break
            for y in range(-R2, R2+1):
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
    
    # Strategy 3: If still at 5, try a different base set or random search for 6
    if best_n == 5 and time.time() < deadline - 10:
        # Try random search with a different seed for 6 points
        random.seed(12345)
        for _ in range(200000):
            if time.time() > deadline:
                break
            pts = []
            for _ in range(6):
                x = random.randint(-150, 150)
                y = random.randint(-150, 150)
                pts.append((x, y))
            pts = list(set(pts))
            if len(pts) < 6:
                continue
            ok = True
            for i in range(6):
                for j in range(i+1, 6):
                    dx = pts[i][0] - pts[j][0]
                    dy = pts[i][1] - pts[j][1]
                    d2 = dx*dx + dy*dy
                    if not _is_sq(d2):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue
            if _valid(pts):
                best = pts
                best_n = 6
                break
    
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in best]}
# EVOLVE-BLOCK-END
