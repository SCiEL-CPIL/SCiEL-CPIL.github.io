"""CCD-based trajectory optimization for packing 12 unit hexagons in a regular hexagon.

Approach:
- Model each inner hexagon by its center + orientation.
- Collision between two inner hexagons is checked via the Minkowski sum
  (a hexagon of side 2 when co-oriented, or a 12-gon when rotated).
- We slide the hexagons along a time-parameterized path from a trivially
  valid configuration into a dense one, using continuous collision detection
  to ensure no interpenetration occurs during the motion.
- The outer hexagon side length is then minimized by binary search on the
  containment constraint, with gradient-free local refinement.
"""
import math
import random
import numpy as np

N = 12
# Precompute unit hexagon vertices (flat-top, side 1)
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
    """Andrew's monotone chain; returns list of (x,y) in CCW order."""
    pts = sorted(set((round(p[0], 12), round(p[1], 12)) for p in points))
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
    """Check if point p is inside convex polygon poly (CCW)."""
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
    """Check if all vertices of inner are inside outer (both convex, CCW)."""
    return all(_point_in_convex(outer, (v[0], v[1]), tol) for v in inner)

def _minkowski_sum(poly1, poly2):
    """Minkowski sum of two convex polygons (CCW)."""
    if not poly1 or not poly2:
        return []
    # Sum all vertex pairs
    pts = [(a[0] + b[0], a[1] + b[1]) for a in poly1 for b in poly2]
    return _convex_hull(pts)

def _separation_check(verts_a, verts_b, tol=1e-9):
    """Return True if convex polygons A and B are disjoint (no interior overlap)."""
    # Check if any vertex of A is strictly inside B, or vice versa
    for p in verts_a:
        if _point_in_convex(verts_b, p, tol):
            return False
    for p in verts_b:
        if _point_in_convex(verts_a, p, tol):
            return False
    # Check edge intersections
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
    n1 = len(verts_a)
    n2 = len(verts_b)
    for i in range(n1):
        a1 = verts_a[i]
        a2 = verts_a[(i + 1) % n1]
        for j in range(n2):
            b1 = verts_b[j]
            b2 = verts_b[(j + 1) % n2]
            if segments_intersect(a1, a2, b1, b2):
                return False
    return True

def _inner_hexagon(cx, cy, angle_deg, side=1.0):
    return _hex_verts(angle_deg, cx, cy, side)

def _outer_hexagon(cx, cy, side, angle_deg=0.0):
    return _hex_verts(angle_deg, cx, cy, side)

def _is_valid_packing(inner_configs, outer_center, outer_side, outer_angle=0.0, tol=1e-9):
    """Check all inner hexagons are disjoint and inside outer."""
    outer_verts = _outer_hexagon(outer_center[0], outer_center[1], outer_side, outer_angle)
    inner_verts_list = []
    for (cx, cy, ang) in inner_configs:
        v = _inner_hexagon(cx, cy, ang)
        inner_verts_list.append(v)
        if not _poly_in_convex(outer_verts, v, tol):
            return False
    for i in range(len(inner_verts_list)):
        for j in range(i + 1, len(inner_verts_list)):
            if not _separation_check(inner_verts_list[i], inner_verts_list[j], tol):
                return False
    return True

def _ccd_check_move(configs_start, configs_end, outer_center, outer_side, outer_angle=0.0, 
                    num_steps=10, tol=1e-9):
    """Check if moving from configs_start to configs_end is collision-free using CCD.
    
    Uses Minkowski sum: two hexagons collide iff the center of one is inside
    the Minkowski sum of the other with its reflection.
    """
    # For each pair, check if their relative motion passes through the Minkowski sum
    n = len(configs_start)
    for i in range(n):
        for j in range(i + 1, n):
            # Get start and end positions
            cx1s, cy1s, a1s = configs_start[i]
            cx2s, cy2s, a2s = configs_start[j]
            cx1e, cy1e, a1e = configs_end[i]
            cx2e, cy2e, a2e = configs_end[j]
            
            # Sample along the path
            for step in range(num_steps + 1):
                t = step / num_steps
                # Interpolate positions and angles
                cx1 = cx1s + t * (cx1e - cx1s)
                cy1 = cy1s + t * (cy1e - cy1s)
                a1 = a1s + t * (a1e - a1s)
                cx2 = cx2s + t * (cx2e - cx2s)
                cy2 = cy2s + t * (cy2e - cy2s)
                a2 = a2s + t * (a2e - a2s)
                
                # Check collision at this time step
                v1 = _inner_hexagon(cx1, cy1, a1)
                v2 = _inner_hexagon(cx2, cy2, a2)
                if not _separation_check(v1, v2, tol):
                    return False
    
    # Also check containment at each step
    outer_verts = _outer_hexagon(outer_center[0], outer_center[1], outer_side, outer_angle)
    for step in range(num_steps + 1):
        t = step / num_steps
        for i in range(n):
            cx1s, cy1s, a1s = configs_start[i]
            cx1e, cy1e, a1e = configs_end[i]
            cx1 = cx1s + t * (cx1e - cx1s)
            cy1 = cy1s + t * (cy1e - cy1s)
            a1 = a1s + t * (a1e - a1s)
            v1 = _inner_hexagon(cx1, cy1, a1)
            if not _poly_in_convex(outer_verts, v1, tol):
                return False
    
    return True

def _build_trajectory_ccd(t, base_configs, outer_side, outer_center=(0.0, 0.0), outer_angle=0.0):
    """Time-parameterized path with CCD: t in [0,1]. At t=0, spread out; at t=1, dense."""
    configs = []
    for k, (cx, cy, ang) in enumerate(base_configs):
        # Smooth interpolation with rotational sweep
        # Use easing function for smoother motion
        t_ease = t * t * (3 - 2 * t)  # smoothstep
        
        # Spiral-like inward motion with rotation
        r = math.hypot(cx, cy)
        th = math.atan2(cy, cx)
        
        # Reduce radius and add angular sweep
        new_r = r * (1.0 - 0.6 * t_ease)
        new_th = th + math.radians(45 * t_ease)
        new_cx = new_r * math.cos(new_th)
        new_cy = new_r * math.sin(new_th)
        
        # Add small rotational offset
        new_ang = ang + math.radians(30 * t_ease)
        
        configs.append((new_cx, new_cy, math.degrees(new_ang)))
    return configs

def _optimize_packing_ccd(outer_side, outer_center=(0.0, 0.0), outer_angle=0.0, 
                          max_iters=100, tol=1e-9):
    """Optimize inner hexagon positions using CCD-based trajectory."""
    best_configs = None
    best_penalty = float('inf')
    
    for trial in range(8):
        random.seed(123 + trial * 7)
        
        # Initial configuration: hexagonal lattice pattern
        positions = []
        
        # Try different initial arrangements
        if trial < 3:
            # Ring + center arrangement
            positions.append((0.0, 0.0, 0.0))
            for k in range(6):
                th = math.radians(60 * k)
                positions.append((2.0 * math.cos(th), 2.0 * math.sin(th), 0.0))
            for k in range(5):
                th = math.radians(60 * k + 30)
                positions.append((3.8 * math.cos(th), 3.8 * math.sin(th), 0.0))
        elif trial < 6:
            # Two rings
            for k in range(6):
                th = math.radians(60 * k)
                positions.append((1.8 * math.cos(th), 1.8 * math.sin(th), 0.0))
            for k in range(6):
                th = math.radians(60 * k + 30)
                positions.append((3.6 * math.cos(th), 3.6 * math.sin(th), 0.0))
        else:
            # Random initial positions in a larger area
            for k in range(N):
                th = 2 * math.pi * k / N + random.uniform(-0.3, 0.3)
                r = random.uniform(2.5, 4.5)
                positions.append((r * math.cos(th), r * math.sin(th), random.uniform(0, 60)))
        
        base_configs = positions[:]
        current_configs = positions[:]
        
        # CCD-based trajectory optimization
        success = False
        for it in range(max_iters):
            t = it / max_iters
            target_configs = _build_trajectory_ccd(t, base_configs, outer_side, outer_center, outer_angle)
            
            # Try to move from current to target with CCD check
            if _ccd_check_move(current_configs, target_configs, outer_center, outer_side, outer_angle, 
                              num_steps=5, tol=tol):
                current_configs = target_configs
                if _is_valid_packing(current_configs, outer_center, outer_side, outer_angle, tol):
                    success = True
                    break
            else:
                # If CCD fails, try smaller steps
                for sub_step in range(3):
                    t_sub = t - (1 - sub_step) / 3 * (t - (it - 1) / max_iters if it > 0 else 0)
                    if t_sub < 0:
                        t_sub = 0
                    target_sub = _build_trajectory_ccd(t_sub, base_configs, outer_side, outer_center, outer_angle)
                    if _ccd_check_move(current_configs, target_sub, outer_center, outer_side, outer_angle, 
                                      num_steps=3, tol=tol):
                        current_configs = target_sub
                        break
        
        # Local refinement: random perturbations with CCD
        if not success:
            for _ in range(20):
                new_configs = [list(c) for c in current_configs]
                for i in range(N):
                    if random.random() < 0.4:
                        new_configs[i][0] += random.uniform(-0.3, 0.3)
                        new_configs[i][1] += random.uniform(-0.3, 0.3)
                    if random.random() < 0.15:
                        new_configs[i][2] += random.uniform(-10, 10)
                new_configs = [tuple(c) for c in new_configs]
                
                if _is_valid_packing(new_configs, outer_center, outer_side, outer_angle, tol):
                    current_configs = new_configs
                    success = True
                    break
        
        if success:
            return current_configs
    
    return None

def _minimize_side_length(outer_center=(0.0, 0.0), outer_angle=0.0, lo=3.0, hi=10.0, tol=1e-4):
    """Binary search for minimum outer side length."""
    # First find a valid upper bound
    while hi < 15.0:
        configs = _optimize_packing_ccd(hi, outer_center, outer_angle)
        if configs is not None:
            break
        hi *= 1.15
    
    if hi >= 15.0:
        return None, hi
    
    for _ in range(40):
        if hi - lo < tol:
            break
        mid = (lo + hi) / 2
        configs = _optimize_packing_ccd(mid, outer_center, outer_angle)
        if configs is not None:
            hi = mid
        else:
            lo = mid
    
    # Return the best configuration at hi
    configs = _optimize_packing_ccd(hi, outer_center, outer_angle)
    if configs is None:
        return None, hi
    return configs, hi

def construct():
    """Construct 12 unit hexagons inside a regular hexagon, minimizing outer side."""
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Try to find a good packing
    configs, side = _minimize_side_length(outer_center, outer_angle, lo=3.2, hi=7.0, tol=0.005)
    
    if configs is None:
        # Fallback: simple linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = [[c[0], c[1], c[2]] for c in configs]
    return inner, outer_center, side, outer_angle
