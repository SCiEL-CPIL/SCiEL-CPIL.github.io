import time
from math import isqrt

_START = None

def _elapsed():
    global _START
    if _START is None:
        import time as _t
        _START = _t.time()
    return time.time() - _START

def _deadline_check(limit=140.0):
    return _elapsed() < limit

def _is_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def _dist_sq(p1, p2):
    return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

def _are_collinear(p1, p2, p3):
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]) == 0

def _are_concyclic(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    a1 = x1 - x2
    b1 = y1 - y2
    c1 = x2*x2 + y2*y2 - x1*x1 - y1*y1
    
    a2 = x1 - x3
    b2 = y1 - y3
    c2 = x3*x3 + y3*y3 - x1*x1 - y1*y1
    
    det = a1 * b2 - a2 * b1
    if det == 0:
        return False
    
    D_num = c1 * b2 - c2 * b1
    E_num = a1 * c2 - a2 * c1
    
    val = (det * (x4*x4 + y4*y4) + D_num * x4 + E_num * y4 
           - det * (x1*x1 + y1*y1) - D_num * x1 - E_num * y1)
    return val == 0

def _valid_set(points):
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dsq = _dist_sq(points[i], points[j])
            if dsq == 0:
                return False
            if not _is_square(dsq):
                return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if _are_collinear(points[i], points[j], points[k]):
                    return False
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                for l in range(k + 1, n):
                    if _are_concyclic(points[i], points[j], points[k], points[l]):
                        return False
    return True

def _circle_coeffs(p1, p2, p3):
    """Return (det, D_num, E_num) for circle x^2+y^2 + (D_num/det)x + (E_num/det)y + F = 0
    such that F = -(det*(x1^2+y1^2) + D_num*x1 + E_num*y1) / det
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    a1 = x1 - x2
    b1 = y1 - y2
    c1 = x2*x2 + y2*y2 - x1*x1 - y1*y1
    
    a2 = x1 - x3
    b2 = y1 - y3
    c2 = x3*x3 + y3*y3 - x1*x1 - y1*y1
    
    det = a1 * b2 - a2 * b1
    if det == 0:
        return None
    
    D_num = c1 * b2 - c2 * b1
    E_num = a1 * c2 - a2 * c1
    return (det, D_num, E_num)

def construct():
    global _START
    _START = time.time()
    deadline = 138.0
    
    # Known 5-point set
    fallback = [(0, 0), (-56, 33), (-16, 63), (0, 33), (16, 63)]
    best = fallback
    best_size = 5
    
    # Strategy 1: Brute force search for a 6th point near the 5-point set
    found_6 = False
    for x in range(-150, 151):
        if not _deadline_check(deadline):
            break
        for y in range(-30, 151):
            if not _deadline_check(deadline):
                break
            p = (x, y)
            ok = True
            for sp in fallback:
                dsq = _dist_sq(p, sp)
                if dsq == 0 or not _is_square(dsq):
                    ok = False
                    break
            if not ok:
                continue
            
            new_pts = fallback + [p]
            ok = True
            for i in range(6):
                if not ok: break
                for j in range(i+1, 6):
                    if not ok: break
                    for k in range(j+1, 6):
                        if _are_collinear(new_pts[i], new_pts[j], new_pts[k]):
                            ok = False
                            break
            if not ok:
                continue
            
            ok = True
            for i in range(6):
                if not ok: break
                for j in range(i+1, 6):
                    if not ok: break
                    for k in range(j+1, 6):
                        if not ok: break
                        for l in range(k+1, 6):
                            if _are_concyclic(new_pts[i], new_pts[j], new_pts[k], new_pts[l]):
                                ok = False
                                break
            if ok:
                best = new_pts
                best_size = 6
                found_6 = True
                break
        if found_6:
            break
    
    # Strategy 2: If 6 found, try to find a 7th point
    if best_size == 6:
        seed6 = best
        found_7 = False
        for x in range(-100, 101):
            if not _deadline_check(deadline):
                break
            for y in range(-50, 101):
                if not _deadline_check(deadline):
                    break
                p = (x, y)
                ok = True
                for sp in seed6:
                    dsq = _dist_sq(p, sp)
                    if dsq == 0 or not _is_square(dsq):
                        ok = False
                        break
                if not ok:
                    continue
                
                new_pts = seed6 + [p]
                ok = True
                for i in range(7):
                    if not ok: break
                    for j in range(i+1, 7):
                        if not ok: break
                        for k in range(j+1, 7):
                            if _are_collinear(new_pts[i], new_pts[j], new_pts[k]):
                                ok = False
                                break
                if not ok:
                    continue
                
                ok = True
                for i in range(7):
                    if not ok: break
                    for j in range(i+1, 7):
                        if not ok: break
                        for k in range(j+1, 7):
                            if not ok: break
                            for l in range(k+1, 7):
                                if _are_concyclic(new_pts[i], new_pts[j], new_pts[k], new_pts[l]):
                                    ok = False
                                    break
                if ok:
                    best = new_pts
                    best_size = 7
                    found_7 = True
                    break
            if found_7:
                break
    
    # Strategy 3: Try a different known 5-point set if we didn't find 6
    if best_size == 5:
        # Try another 5-point configuration
        # The set: (0,0), (9,12), (0,15), (9,27), (18,24) - check if valid
        alt5 = [(0, 0), (9, 12), (0, 15), (9, 27), (18, 24)]
        if _valid_set(alt5):
            best = alt5
            # Try to add 6th
            for x in range(-50, 51):
                if not _deadline_check(deadline):
                    break
                for y in range(-20, 80):
                    if not _deadline_check(deadline):
                        break
                    p = (x, y)
                    ok = True
                    for sp in best:
                        dsq = _dist_sq(p, sp)
                        if dsq == 0 or not _is_square(dsq):
                            ok = False
                            break
                    if not ok:
                        continue
                    
                    new_pts = best + [p]
                    ok = True
                    for i in range(6):
                        if not ok: break
                        for j in range(i+1, 6):
                            if not ok: break
                            for k in range(j+1, 6):
                                if _are_collinear(new_pts[i], new_pts[j], new_pts[k]):
                                    ok = False
                                    break
                    if not ok:
                        continue
                    
                    ok = True
                    for i in range(6):
                        if not ok: break
                        for j in range(i+1, 6):
                            if not ok: break
                            for k in range(j+1, 6):
                                if not ok: break
                                for l in range(k+1, 6):
                                    if _are_concyclic(new_pts[i], new_pts[j], new_pts[k], new_pts[l]):
                                        ok = False
                                        break
                    if ok:
                        best = new_pts
                        best_size = 6
                        break
                if best_size == 6:
                    break
    
    D = 1
    points_out = [[[x, 1], [y, 1]] for x, y in best]
    return {'D': D, 'points': points_out}
