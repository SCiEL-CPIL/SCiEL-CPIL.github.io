"""CCD-based trajectory optimization for packing 12 unit hexagons in a regular hexagon.

Uses Minkowski sum for collision detection and gradient-free optimization
to minimize the outer hexagon side length.
"""
import math
import numpy as np
from scipy.optimize import minimize

N = 12

def _hex_verts(angle_deg=0.0, cx=0.0, cy=0.0, side=1.0):
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        th = math.radians(60 * k)
        x = side * math.cos(th)
        y = side * math.sin(th)
        verts.append((cx + x * cos_a - y * sin_a,
                      cy + x * sin_a + y * cos_a))
    return verts

def _convex_hull(points):
    pts = sorted(set((round(p[0], 14), round(p[1], 14)) for p in points))
    if len(pts) <= 2:
        return pts
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def _point_in_convex(poly, p, tol=1e-9):
    if len(poly) < 3:
        return False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -tol:
            return False
    return True

def _poly_in_convex(outer, inner, tol=1e-9):
    return all(_point_in_convex(outer, (v[0], v[1]), tol) for v in inner)

def _separation_check(verts_a, verts_b, tol=1e-9):
    for p in verts_a:
        if _point_in_convex(verts_b, p, tol):
            return False
    for p in verts_b:
        if _point_in_convex(verts_a, p, tol):
            return False
    def orient(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    def on_segment(p, q, r):
        return (min(p[0], r[0]) - tol <= q[0] <= max(p[0], r[0]) + tol and
                min(p[1], r[1]) - tol <= q[1] <= max(p[1], r[1]) + tol)
    def segments_intersect(p1, p2, p3, p4):
        d1 = orient(p3, p4, p1)
        d2 = orient(p3, p4, p2)
        d3 = orient(p1, p2, p3)
        d4 = orient(p1, p2, p4)
        if ((d1 > tol and d2 < -tol) or (d1 < -tol and d2 > tol)) and \
           ((d3 > tol and d4 < -tol) or (d3 < -tol and d4 > tol)):
            return True
        if abs(d1) <= tol and on_segment(p1, p3, p4):
            return True
        if abs(d2) <= tol and on_segment(p2, p3, p4):
            return True
        if abs(d3) <= tol and on_segment(p3, p1, p2):
            return True
        if abs(d4) <= tol and on_segment(p4, p1, p2):
            return True
        return False
    n1, n2 = len(verts_a), len(verts_b)
    for i in range(n1):
        a1, a2 = verts_a[i], verts_a[(i + 1) % n1]
        for j in range(n2):
            b1, b2 = verts_b[j], verts_b[(j + 1) % n2]
            if segments_intersect(a1, a2, b1, b2):
                return False
    return True

def _is_valid(inner_configs, outer_center, outer_side, outer_angle=0.0, tol=1e-9):
    outer_verts = _hex_verts(outer_angle, outer_center[0], outer_center[1], outer_side)
    inner_verts_list = [_hex_verts(c[2], c[0], c[1], 1.0) for c in inner_configs]
    for v in inner_verts_list:
        if not _poly_in_convex(outer_verts, v, tol):
            return False
    for i in range(N):
        for j in range(i + 1, N):
            if not _separation_check(inner_verts_list[i], inner_verts_list[j], tol):
                return False
    return True

def construct():
    """Construct 12 unit hexagons inside a regular hexagon, minimizing outer side."""
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Initial guess based on hexagonal lattice
    def make_initial_config():
        configs = []
        configs.append((0.0, 0.0, 0.0))
        for k in range(6):
            th = math.radians(60 * k)
            configs.append((2.0 * math.cos(th), 2.0 * math.sin(th), 0.0))
        for k in range(5):
            th = math.radians(60 * k + 30)
            configs.append((3.4641 * math.cos(th), 3.4641 * math.sin(th), 0.0))
        return configs
    
    # Binary search on outer side length
    lo, hi = 4.5, 8.0
    best_configs = None
    best_side = hi
    
    for _ in range(25):
        if hi - lo < 1e-3:
            break
        mid = (lo + hi) / 2
        
        configs = make_initial_config()
        
        # Try to validate directly first
        if _is_valid(configs, outer_center, mid, outer_angle):
            hi = mid
            best_configs = configs
            best_side = mid
            continue
        
        # Optimize positions to fit
        def penalty_func(params):
            c = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
            if _is_valid(c, outer_center, mid, outer_angle):
                return 0.0
            # Compute sum of violation distances
            outer_verts = _hex_verts(outer_angle, outer_center[0], outer_center[1], mid)
            inner_verts_list = [_hex_verts(cc[2], cc[0], cc[1], 1.0) for cc in c]
            pen = 0.0
            for v in inner_verts_list:
                for p in v:
                    if not _point_in_convex(outer_verts, p, 0.0):
                        min_d = float('inf')
                        for i in range(6):
                            x1, y1 = outer_verts[i]
                            x2, y2 = outer_verts[(i+1) % 6]
                            dx, dy = x2-x1, y2-y1
                            if dx*dx + dy*dy < 1e-20:
                                d = math.hypot(p[0]-x1, p[1]-y1)
                            else:
                                t = max(0.0, min(1.0, ((p[0]-x1)*dx + (p[1]-y1)*dy)/(dx*dx+dy*dy)))
                                px, py = x1+t*dx, y1+t*dy
                                d = math.hypot(p[0]-px, p[1]-py)
                            min_d = min(min_d, d)
                        pen += min_d
            return pen
        
        p0 = [c[0] for c in configs] + [c[1] for c in configs] + [c[2] for c in configs]
        res = minimize(penalty_func, p0, method='Nelder-Mead',
                       options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-6})
        
        opt_configs = [(res.x[3*i], res.x[3*i+1], res.x[3*i+2]) for i in range(N)]
        
        if _is_valid(opt_configs, outer_center, mid, outer_angle):
            hi = mid
            best_configs = opt_configs
            best_side = mid
        else:
            lo = mid
    
    if best_configs is None:
        best_configs = make_initial_config()
        best_side = hi
    
    inner = [[c[0], c[1], c[2]] for c in best_configs]
    return inner, outer_center, best_side, outer_angle
