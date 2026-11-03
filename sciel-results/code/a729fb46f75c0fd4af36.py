"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import differential_evolution

N = 11

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side s, rotated by angle_deg."""
    a = math.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, ... 300 degrees (pointy-top or flat-top depending)
    # Standard: vertices at angle = a + i*pi/3
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4 (including endpoints)."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    if d1 == 0 and _on_segment(p3, p1, p4): return True
    if d2 == 0 and _on_segment(p3, p2, p4): return True
    if d3 == 0 and _on_segment(p1, p3, p2): return True
    if d4 == 0 and _on_segment(p1, p4, p2): return True
    return False

def _on_segment(a, b, c):
    """Check if b is on segment a-c (collinear assumed)."""
    return (min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and
            min(a[1], c[1]) <= b[1] <= max(a[1], c[1]))

def _polygons_overlap(v1, v2, eps=1e-10):
    """Check if two convex polygons (lists of 6 vertices) overlap.
    Returns True if they overlap (including touching with eps margin)."""
    n1 = len(v1)
    n2 = len(v2)
    
    # Check all edge pairs for intersection
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            p3 = v2[j]
            p4 = v2[(j + 1) % n2]
            if _segments_intersect(p1, p2, p3, p4):
                return True
    
    # Check if one vertex of v1 is inside v2
    if _point_in_convex_polygon(v1[0], v2, eps):
        return True
    # Check if one vertex of v2 is inside v1
    if _point_in_convex_polygon(v2[0], v1, eps):
        return True
    
    return False

def _point_in_convex_polygon(p, poly, eps=1e-10):
    """Check if point p is inside convex polygon poly."""
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        # Cross product to check if p is on the 'inside' side of edge a-b
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if cross < -eps:
            return False
    return True

def _min_distance_between_polygons(v1, v2):
    """Compute minimum distance between two convex polygons. If overlapping, return negative."""
    # First check if they overlap
    if _polygons_overlap(v1, v2):
        # Estimate penetration depth roughly
        # Find closest pair of points
        min_d = float('inf')
        for p in v1:
            for q in v2:
                d = math.hypot(p[0] - q[0], p[1] - q[1])
                if d < min_d:
                    min_d = d
        # If centers are very close, likely deep overlap
        c1 = np.mean(np.array(v1), axis=0)
        c2 = np.mean(np.array(v2), axis=0)
        center_dist = np.linalg.norm(c1 - c2)
        return -center_dist  # rough negative distance
    
    # Compute minimum distance between edges
    min_d = float('inf')
    n1 = len(v1)
    n2 = len(v2)
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        for j in range(n2):
            p3 = v2[j]
            p4 = v2[(j + 1) % n2]
            d = _seg_seg_distance(p1, p2, p3, p4)
            if d < min_d:
                min_d = d
    return min_d

def _seg_seg_distance(p1, p2, p3, p4):
    """Minimum distance between two segments."""
    # Check if segments intersect
    if _segments_intersect(p1, p2, p3, p4):
        return 0.0
    # Point-to-segment distances
    d = min(_point_seg_dist(p1, p3, p4), _point_seg_dist(p2, p3, p4),
            _point_seg_dist(p3, p1, p2), _point_seg_dist(p4, p1, p2))
    return d

def _point_seg_dist(p, a, b):
    """Distance from point p to segment a-b."""
    ab = np.array([b[0] - a[0], b[1] - a[1]])
    ap = np.array([p[0] - a[0], p[1] - a[1]])
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-15:
        return np.linalg.norm(ap)
    t = np.dot(ap, ab) / ab_len_sq
    t = max(0.0, min(1.0, t))
    proj = np.array([a[0] + t * ab[0], a[1] + t * ab[1]])
    return np.linalg.norm(np.array(p) - proj)

def _check_feasibility(inner, outer_center, outer_side, outer_angle, gap=1.001):
    """Check if the configuration is feasible (no overlaps, all inside).
    Returns (is_feasible, penalty)."""
    inner_verts = []
    for (x, y, ang) in inner:
        v = _hex_vertices(x, y, 1.0, ang)
        inner_verts.append(v)
    
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    penalty = 0.0
    is_feasible = True
    
    # Check inner-inner overlaps
    for i in range(len(inner_verts)):
        for j in range(i + 1, len(inner_verts)):
            if _polygons_overlap(inner_verts[i], inner_verts[j], gap):
                is_feasible = False
                d = _min_distance_between_polygons(inner_verts[i], inner_verts[j])
                penalty += max(0, -d)
    
    # Check inner-outer containment
    for i, iv in enumerate(inner_verts):
        # All vertices of inner must be inside outer
        for v in iv:
            if not _point_in_convex_polygon(v, outer_verts, -gap):
                is_feasible = False
                # Distance from point to polygon
                min_d = float('inf')
                for k in range(6):
                    a = outer_verts[k]
                    b = outer_verts[(k + 1) % 6]
                    d = _point_seg_dist(v, a, b)
                    if d < min_d:
                        min_d = d
                penalty += min_d
    
    return is_feasible, penalty

def _objective(params):
    """Objective function for optimization. params = [x0,y0,a0, x1,y1,a1, ..., x10,y10,a10, ocx, ocy, oside, oang]"""
    n_inner = 11
    inner = []
    for i in range(n_inner):
        x = params[i * 3]
        y = params[i * 3 + 1]
        a = params[i * 3 + 2]
        inner.append([x, y, a])
    
    ocx = params[n_inner * 3]
    ocy = params[n_inner * 3 + 1]
    oside = params[n_inner * 3 + 2]
    oang = params[n_inner * 3 + 3]
    
    # Penalize large side length
    side_penalty = oside
    
    # Check feasibility
    is_feas, pen = _check_feasibility(inner, (ocx, ocy), oside, oang)
    
    if not is_feas:
        return side_penalty + 100.0 * pen + 10.0
    
    return side_penalty

def _tighten_side(inner, outer_center, outer_angle):
    """Binary search for minimum outer side length that contains all inner hexagons."""
    # Compute bounding box of all inner hexagon vertices
    all_pts = []
    for (x, y, ang) in inner:
        v = _hex_vertices(x, y, 1.0, ang)
        all_pts.extend(v)
    
    if not all_pts:
        return 1.0
    
    # Compute required side length by projecting onto hexagon normals
    # For a regular hexagon with center (cx,cy), side s, angle a,
    # the support function in direction theta is s * max of projections
    # Simpler: try increasing side until all points fit
    
    lo = 1.0
    hi = 20.0
    
    # First, find a feasible hi
    while not _all_inside(inner, outer_center, hi, outer_angle):
        hi *= 2
        if hi > 100:
            break
    
    for _ in range(50):
        mid = (lo + hi) / 2
        if _all_inside(inner, outer_center, mid, outer_angle):
            hi = mid
        else:
            lo = mid
    
    return hi

def _all_inside(inner, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons fit inside outer hexagon."""
    outer_verts = _hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for (x, y, ang) in inner:
        v = _hex_vertices(x, y, 1.0, ang)
        for pt in v:
            if not _point_in_convex_polygon(pt, outer_verts, -1e-9):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s limit, leave margin
    
    # Initial configuration: hexagonal cluster
    # 11 hexagons: 1 center + 6 ring + 4 more
    # For 11 hexagons in a hexagonal arrangement: center + 6 + 4
    # Actually, a hexagonal number: 1, 7, 19... 11 is not a perfect hex number
    # Let's use a compact arrangement
    
    # Initial guess: place hexagons in a grid-like pattern
    inner_init = []
    # Center
    inner_init.append([0.0, 0.0, 0.0])
    # 6 around center (distance = 2 * apothem = 2 * sqrt(3)/2 * 1 = sqrt(3) for touching)
    # For unit hexagons, touching distance between centers = 2 * (sqrt(3)/2) = sqrt(3)
    for i in range(6):
        ang = i * 60.0
        d = math.sqrt(3)
        x = d * math.cos(math.radians(ang))
        y = d * math.sin(math.radians(ang))
        inner_init.append([x, y, 0.0])
    # 4 more: place at next ring positions
    # Ring 2 positions (partial)
    for i in range(4):
        ang = i * 60.0 + 30.0
        d = 2 * math.sqrt(3)
        x = d * math.cos(math.radians(ang))
        y = d * math.sin(math.radians(ang))
        inner_init.append([x, y, 0.0])
    
    # Initial outer
    ocx, ocy = 0.0, 0.0
    oside = 10.0
    oang = 0.0
    
    # Define bounds for differential evolution
    # Inner positions: roughly [-5, 5]
    # Angles: [0, 360]
    # Outer center: [-5, 5]
    # Outer side: [3, 15]
    # Outer angle: [0, 360]
    
    bounds = []
    for _ in range(11):
        bounds.append((-5.0, 5.0))  # x
        bounds.append((-5.0, 5.0))  # y
        bounds.append((0.0, 360.0))  # angle
    bounds.append((-5.0, 5.0))  # outer cx
    bounds.append((-5.0, 5.0))  # outer cy
    bounds.append((3.0, 15.0))  # outer side
    bounds.append((0.0, 360.0))  # outer angle
    
    # Use differential evolution with time limit
    try:
        result = differential_evolution(
            _objective,
            bounds,
            maxiter=200,
            popsize=15,
            tol=1e-4,
            seed=42,
            polish=False,
            callback=lambda xk, convergence: time.time() > deadline
        )
        
        if result.success or result.fun < float('inf'):
            best_params = result.x
            inner_best = []
            for i in range(11):
                inner_best.append([best_params[i*3], best_params[i*3+1], best_params[i*3+2]])
            ocx_best = best_params[33]
            ocy_best = best_params[34]
            oside_best = best_params[35]
            oang_best = best_params[36]
            
            # Tighten the outer side
            oside_tight = _tighten_side(inner_best, (ocx_best, ocy_best), oang_best)
            
            # Verify feasibility with tightened side
            is_feas, pen = _check_feasibility(inner_best, (ocx_best, ocy_best), oside_tight, oang_best)
            if is_feas:
                return inner_best, (ocx_best, ocy_best), oside_tight, oang_best
            else:
                # Use the DE result side if tighter one fails
                return inner_best, (ocx_best, ocy_best), oside_best, oang_best
        
    except Exception:
        pass
    
    # Fallback: use initial configuration with computed side
    oside_init = _tighten_side(inner_init, (ocx, ocy), oang)
    is_feas, pen = _check_feasibility(inner_init, (ocx, ocy), oside_init, oang)
    if is_feas:
        return inner_init, (ocx, ocy), oside_init, oang
    
    # Last resort: return a safe large container
    return inner_init, (0.0, 0.0), 15.0, 0.0

# EVOLVE-BLOCK-END
