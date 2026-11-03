"""Optimized hexagon packing: 11 unit hexagons in a compact hexagonal lattice."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

N = 11

def _soft_min(x, beta=200.0):
    """Smooth approximation of min(x) using log-sum-exp."""
    x_min = np.min(x)
    return x_min - np.log(np.sum(np.exp(-beta * (x - x_min)))) / beta

def _smooth_outer_side(hexes, oangle_rad, beta=200.0):
    """
    Differentiable approximation of the minimum outer hexagon side length.
    Uses soft-min to smooth the nonsupport function min over faces.
    Returns (side, center_x, center_y) where side is the smoothed value.
    """
    points = []
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        points.extend(verts)
    
    if not points:
        return 0.0, 0.0, 0.0
    
    pts = np.array(points)
    
    u_vectors = []
    for k in range(6):
        theta = oangle_rad + math.radians(30 + 60 * k)
        u_vectors.append((math.cos(theta), math.sin(theta)))
    
    # For each face direction, compute the projection range
    # The apothem needed is (max_proj - min_proj) / 2
    # The center is (max_proj + min_proj) / 2 projected back
    # Side = apothem * 2 / sqrt(3)
    
    apothems = np.zeros(6)
    centers_proj = np.zeros(6)
    
    for k in range(6):
        ux, uy = u_vectors[k]
        projections = pts[:, 0] * ux + pts[:, 1] * uy
        apothems[k] = (np.max(projections) - np.min(projections)) / 2.0
        centers_proj[k] = (np.max(projections) + np.min(projections)) / 2.0
    
    # The side length is determined by the maximum apothem needed
    # Smooth this with soft-max
    beta_max = 200.0
    apothem_max = np.max(apothems)
    smooth_apothem = apothem_max + np.log(np.sum(np.exp(beta_max * (apothems - apothem_max)))) / beta_max
    
    # For center, use the face that determines the side
    # Weighted average of center projections
    weights = np.exp(beta_max * (apothems - apothem_max))
    weights /= np.sum(weights)
    
    # Reconstruct center from weighted average of projections
    # This is an approximation; for exact center we'd need the LP
    cx_est = 0.0
    cy_est = 0.0
    for k in range(6):
        ux, uy = u_vectors[k]
        cx_est += weights[k] * centers_proj[k] * ux
        cy_est += weights[k] * centers_proj[k] * uy
    
    side = smooth_apothem * 2.0 / math.sqrt(3)
    return side, cx_est, cy_est

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_deg degrees."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side `side` rotated by angle_deg."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 properly intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False

def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons have disjoint interiors."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    r1 = s1
    r2 = s2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (r1 + r2) ** 2 + 1e-12:
        return True
    verts1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    verts2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        v1a = verts1[i]
        v1b = verts1[(i + 1) % 6]
        for j in range(6):
            v2a = verts2[j]
            v2b = verts2[(j + 1) % 6]
            if _segments_intersect(v1a, v1b, v2a, v2b):
                return False
    return True

def _all_disjoint(inner_hexes, eps=1e-9):
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True

def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True

def _min_outer_side_analytical(hexes, oangle_rad):
    """
    Find minimum side length of outer hexagon with rotation oangle_rad
    that contains all hexagons in `hexes` (centered at origin).
    Returns (min_side, center_x, center_y)
    """
    points = []
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        points.extend(verts)
    
    if not points:
        return 0.0, 0.0, 0.0
    
    pts = np.array(points)
    
    u_vectors = []
    for k in range(6):
        theta = oangle_rad + math.radians(30 + 60 * k)
        u_vectors.append((math.cos(theta), math.sin(theta)))
    
    P_vals = np.zeros(6)
    Q_vals = np.zeros(6)
    for k in range(6):
        ux, uy = u_vectors[k]
        projections = pts[:, 0] * ux + pts[:, 1] * uy
        P_vals[k] = np.max(projections)
        Q_vals[k] = np.min(projections)
    
    A_mat = np.zeros((12, 3))
    b_vec = np.zeros(12)
    for k in range(6):
        ux, uy = u_vectors[k]
        A_mat[2*k, 0] = ux
        A_mat[2*k, 1] = uy
        A_mat[2*k, 2] = -1
        b_vec[2*k] = Q_vals[k]
        
        A_mat[2*k+1, 0] = -ux
        A_mat[2*k+1, 1] = -uy
        A_mat[2*k+1, 2] = -1
        b_vec[2*k+1] = -P_vals[k]
    
    c_vec = np.array([0.0, 0.0, 1.0])
    
    try:
        result = linprog(c_vec, A_ub=A_mat, b_ub=b_vec, bounds=[(None, None), (None, None), (0, None)], method='highs')
        if result.success:
            A_opt = result.x[2]
            ox_opt = result.x[0]
            oy_opt = result.x[1]
            side = A_opt * 2.0 / math.sqrt(3)
            return side, ox_opt, oy_opt
    except Exception:
        pass
    
    return float('inf'), 0.0, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    sqrt3 = math.sqrt(3)
    GAP = 1.0

    def axial_to_xy(q, r):
        x = 1.5 * q * GAP
        y = sqrt3 * (r + q / 2.0) * GAP
        return (x, y)

    pos_axial = [(0, 0)]
    pos_axial.extend([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
    
    ring2_candidates = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), 
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), 
        (0, 2), (1, 2)
    ]
    
    best_side = float('inf')
    best_angle = 0.0
    best_outer_center = (0.0, 0.0)
    best_xy = None

    all_axial = pos_axial + ring2_candidates
    all_xy = [axial_to_xy(q, r) for q, r in all_axial]
    ring2_indices = list(range(7, len(all_xy)))

    # Phase 1: Coarse screening with exact LP at a few angles
    # Use symmetry: only need to check angles 0-30 degrees
    # For each combo, do a quick 3-angle scan then golden section
    for combo in itertools.combinations(ring2_indices, 4):
        if time.time() > deadline - 20.0:
            break
        
        selected_indices = list(range(7)) + list(combo)
        full_xy = [all_xy[i] for i in selected_indices]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        # Quick 3-point scan to find approximate best angle
        scan_angles = [0.0, 15.0, 30.0]
        best_scan_angle = 0.0
        best_scan_side = float('inf')
        
        for ang_deg in scan_angles:
            s = _min_outer_side_analytical(inner_hexes, math.radians(ang_deg))[0]
            if s < best_scan_side:
                best_scan_side = s
                best_scan_angle = ang_deg
        
        # Golden section refinement around the best scan angle
        lo = max(0.0, best_scan_angle - 5.0)
        hi = min(30.0, best_scan_angle + 5.0)
        
        gr = (math.sqrt(5) - 1) / 2
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        
        fc = _min_outer_side_analytical(inner_hexes, math.radians(c))[0]
        fd = _min_outer_side_analytical(inner_hexes, math.radians(d))[0]
        
        for _ in range(12):
            if time.time() > deadline - 20.0:
                break
            if fc < fd:
                hi = d
                d = c
                fd = fc
                c = hi - gr * (hi - lo)
                fc = _min_outer_side_analytical(inner_hexes, math.radians(c))[0]
            else:
                lo = c
                c = d
                fc = fd
                d = lo + gr * (hi - lo)
                fd = _min_outer_side_analytical(inner_hexes, math.radians(d))[0]
        
        best_angle_candidate = (lo + hi) / 2.0
        side, ox, oy = _min_outer_side_analytical(inner_hexes, math.radians(best_angle_candidate))
        
        if side < best_side:
            best_side = side
            best_angle = best_angle_candidate
            best_xy = full_xy
            best_outer_center = (ox, oy)

    if best_xy is None:
        best_xy = [all_xy[i] for i in range(7)]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]
        best_side, ox, oy = _min_outer_side_analytical(inner_hexes, 0.0)
        best_outer_center = (ox, oy)
        best_angle = 0.0

    # Phase 2: High-precision polish on the best configuration
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]
    
    # Fine golden section search with tighter window
    lo = max(0.0, best_angle - 0.5)
    hi = min(30.0, best_angle + 0.5)
    
    gr = (math.sqrt(5) - 1) / 2
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    
    fc = _min_outer_side_analytical(inner_hexes, math.radians(c))[0]
    fd = _min_outer_side_analytical(inner_hexes, math.radians(d))[0]
    
    polish_steps = 0
    while time.time() < deadline - 3.0 and polish_steps < 60:
        if fc < fd:
            hi = d
            d = c
            fd = fc
            c = hi - gr * (hi - lo)
            fc = _min_outer_side_analytical(inner_hexes, math.radians(c))[0]
        else:
            lo = c
            c = d
            fc = fd
            d = lo + gr * (hi - lo)
            fd = _min_outer_side_analytical(inner_hexes, math.radians(d))[0]
        polish_steps += 1
    
    best_angle_polished = (lo + hi) / 2.0
    side_polished, ox_polished, oy_polished = _min_outer_side_analytical(inner_hexes, math.radians(best_angle_polished))
    
    if side_polished < best_side:
        best_side = side_polished
        best_angle = best_angle_polished
        best_outer_center = (ox_polished, oy_polished)

    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]

    if not _all_disjoint(inner_hexes):
        # With GAP=1.0, hexagons share edges but have disjoint interiors.
        # If numerical issues cause false positives, scale up slightly.
        for extra_gap in [1.001, 1.005, 1.01]:
            inner_hexes = [(x * extra_gap, y * extra_gap, 1.0, 0.0) for x, y in best_xy]
            if _all_disjoint(inner_hexes):
                break

    outer_hex = (best_outer_center[0], best_outer_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        best_side *= 1.01
        outer_hex = (best_outer_center[0], best_outer_center[1], best_side, best_angle)
        
        valid = True
        for h in inner_hexes:
            if not _inner_in_outer(h, outer_hex):
                valid = False
                break

    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_outer_center[0], best_outer_center[1]), best_side, best_angle
