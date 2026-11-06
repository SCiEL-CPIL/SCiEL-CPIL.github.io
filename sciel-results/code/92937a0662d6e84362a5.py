"""Search for integral point sets with rational distances, no 3 collinear, no 4 concyclic."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction
import math

def _is_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _no_three_collinear(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                dx1 = points[j][0] - points[i][0]
                dy1 = points[j][1] - points[i][1]
                dx2 = points[k][0] - points[i][0]
                dy2 = points[k][1] - points[i][1]
                cross = dx1*dy2 - dy1*dx2
                if cross == 0:
                    return False
    return True

def _no_four_concyclic(points):
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for m in range(k+1, n):
                    # Circle equation: x^2 + y^2 + Ax + By + C = 0
                    # Det of [x^2+y^2, x, y, 1] for 4 points
                    rows = []
                    for idx in [i, j, k, m]:
                        x = points[idx][0]
                        y = points[idx][1]
                        rows.append([x*x + y*y, x, y, 1])
                    a = rows[0]; b = rows[1]; c = rows[2]; d = rows[3]
                    det = (a[0]*(b[1]*(c[2]*d[3]-c[3]*d[2]) - b[2]*(c[1]*d[3]-c[3]*d[1]) + b[3]*(c[1]*d[2]-c[2]*d[1]))
                         - a[1]*(b[0]*(c[2]*d[3]-c[3]*d[2]) - b[2]*(c[0]*d[3]-c[3]*d[0]) + b[3]*(c[0]*d[2]-c[2]*d[0]))
                         + a[2]*(b[0]*(c[1]*d[3]-c[3]*d[1]) - b[1]*(c[0]*d[3]-c[3]*d[0]) + b[3]*(c[0]*d[1]-c[1]*d[0]))
                         - a[3]*(b[0]*(c[1]*d[2]-c[2]*d[1]) - b[1]*(c[0]*d[2]-c[2]*d[0]) + b[2]*(c[0]*d[1]-c[1]*d[0])))
                    if det == 0:
                        return False
    return True

def _find_6_point_set(deadline):
    # Base 5 points
    base = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    A = base[0]
    D = base[3]
    
    # We want to find (x, y) such that dist to A and D are integers
    # dist(A, F)^2 = x^2 + y^2 = dA^2
    # dist(D, F)^2 = x^2 + (y-33)^2 = dD^2
    # Subtracting: (y-33)^2 - y^2 = dD^2 - dA^2
    # y^2 - 66y + 33^2 - y^2 = dD^2 - dA^2
    # -66y + 1089 = dD^2 - dA^2
    # 66y = 1089 + dA^2 - dD^2
    # y = (1089 + dA^2 - dD^2) / 66
    
    # Search range for dA, dD
    # Max coordinate in base is ~63. 
    # Let's search dA, dD up to 200.
    limit = 200
    found = None
    
    for dA in range(1, limit):
        for dD in range(1, limit):
            num_y = 1089 + dA*dA - dD*dD
            if num_y % 66 != 0:
                continue
            y = num_y // 66
            # y must be such that x^2 = dA^2 - y^2 is a non-negative perfect square
            x_sq = dA*dA - y*y
            if x_sq < 0:
                continue
            x = math.isqrt(x_sq)
            if x*x != x_sq:
                continue
            
            # Candidates: (x, y) and (-x, y)
            for sign in [1, -1]:
                cx = sign * x
                cy = y
                cand = (cx, cy)
                
                # Check distinct
                if cand in base:
                    continue
                
                # Check distances to other base points B, C, E
                ok = True
                for p in base:
                    if p == A or p == D:
                        continue
                    dx = cx - p[0]
                    dy = cy - p[1]
                    d_sq = dx*dx + dy*dy
                    if not _is_square(d_sq):
                        ok = False
                        break
                
                if not ok:
                    continue
                
                # Full set
                full_set = base + [cand]
                
                # Check general position
                if not _no_three_collinear(full_set):
                    continue
                if not _no_four_concyclic(full_set):
                    continue
                
                found = full_set
                break
            if found:
                break
        if found:
            break
        if time.time() > deadline:
            break
            
    return found

def construct():
    start = time.time()
    deadline = start + 100
    
    result = _find_6_point_set(deadline)
    
    if result:
        return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in result]}
    
    # Fallback to known 5-point set
    pts = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    return {'D': 1, 'points': [[[x, 1], [y, 1]] for x, y in pts]}
# EVOLVE-BLOCK-END
