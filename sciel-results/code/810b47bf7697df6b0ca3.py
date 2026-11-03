"""Direct hexagon packing: 11 unit hexagons in a hexagonal lattice pattern.
Uses LP-based optimization for the outer hexagon parameters."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog

N = 11


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


def _min_outer_side(hexes, ox, oy, oangle=0.0):
    """Find minimum side length of outer hexagon centered at (ox, oy) with rotation oangle
    that contains all hexagons in `hexes`."""
    max_val = 0.0
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                proj = lx * math.cos(theta) + ly * math.sin(theta)
                if proj > max_val:
                    max_val = proj
    return max_val * 2.0 / math.sqrt(3)


def _lp_min_enclosing_hex(inner_points, n_outer_angles=6):
    """
    Find the minimum side length of a regular hexagon that encloses all points.
    The outer hexagon can translate and rotate.
    
    inner_points: list of (x, y) points (vertices of inner hexagons)
    Returns: (min_side, center_x, center_y, angle_rad)
    """
    n_pts = len(inner_points)
    if n_pts == 0:
        return (0.0, 0.0, 0.0, 0.0)
    
    # For each point and each of the 6 hexagon face normals, we have a constraint:
    # The projection of (point - center) onto normal_k <= apothem = side * sqrt(3)/2
    # 
    # Let's parameterize:
    # - center = (cx, cy)
    # - angle = theta (rotation of outer hexagon)
    # - side = s
    #
    # Constraint: for each point p_i and each face normal direction k:
    # (p_i - c) · n_k(theta) <= s * sqrt(3)/2
    #
    # This is nonlinear due to theta. We'll use a grid search over theta,
    # then for each fixed theta, solve an LP for (cx, cy, s).
    
    best_result = None
    best_side = float('inf')
    
    # Grid search over outer hexagon angle
    for theta_deg in np.linspace(0, 60, 121):  # 0 to 60 degrees, 121 points
        theta = math.radians(theta_deg)
        
        # Face normals for a hexagon rotated by theta:
        # Normal directions are at theta + 30 + 60*k for k=0..5
        normals = []
        for k in range(6):
            a = theta + math.radians(30 + 60 * k)
            normals.append((math.cos(a), math.sin(a)))
        
        # For fixed theta, we have a linear program:
        # Variables: cx, cy, s (3 variables)
        # Minimize: s
        # Subject to: for each point i and each normal k:
        #   (x_i - cx) * nx_k + (y_i - cy) * ny_k <= s * sqrt(3)/2
        #   => -x_i*nx_k - y_i*ny_k + cx*nx_k + cy*ny_k - s*sqrt(3)/2 <= 0
        #   => cx*nx_k + cy*ny_k - s*sqrt(3)/2 <= x_i*nx_k + y_i*ny_k
        
        n_constraints = n_pts * 6
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)
        
        idx = 0
        for px, py in inner_points:
            for nx, ny in normals:
                A_ub[idx, 0] = nx
                A_ub[idx, 1] = ny
                A_ub[idx, 2] = -math.sqrt(3) / 2
                b_ub[idx] = px * nx + py * ny
                idx += 1
        
        # Minimize s (index 2)
        c_obj = np.array([0.0, 0.0, 1.0])
        
        # Bounds: s >= 0, cx and cy unbounded (but practically bounded)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                s_val = res.x[2]
                if s_val < best_side - 1e-9:
                    best_side = s_val
                    best_result = (s_val, res.x[0], res.x[1], theta)
        except Exception:
            continue
    
    if best_result is not None:
        return best_result
    
    # Fallback: use bounding box
    xs = [p[0] for p in inner_points]
    ys = [p[1] for p in inner_points]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    max_dist = max(math.sqrt((x - cx)**2 + (y - cy)**2) for x, y in inner_points)
    side = max_dist * 2 / math.sqrt(3)
    return (side, cx, cy, 0.0)


def _nelder_mead(f, x0, bounds=None, max_iter=1000, tol=1e-12):
    """Nelder-Mead optimization. Returns (x_best, f_best)."""
    n = len(x0)
    # Initial simplex
    simplex = [list(x0)]
    for i in range(n):
        x = list(x0)
        step = 0.1 if bounds is None else (bounds[i][1] - bounds[i][0]) * 0.1
        x[i] += step
        simplex.append(x)
    
    fvals = [f(s) for s in simplex]
    
    for it in range(max_iter):
        # Sort by function value
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        
        # Check convergence
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        
        # Centroid of all but worst point
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        
        # Reflection
        alpha = 1.0
        xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
        fr = f(xr)
        
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        elif fr < fvals[0]:
            # Expansion
            gamma = 2.0
            xe = [centroid[j] + gamma * (centroid[j] - simplex[-1][j]) for j in range(n)]
            fe = f(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        else:
            # Contraction
            rho = 0.5
            if fr < fvals[-1]:
                xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
            else:
                xc = [centroid[j] - rho * (centroid[j] - simplex[-1][j]) for j in range(n)]
            fc = f(xc)
            if fc < min(fr, fvals[-1]):
                simplex[-1] = xc
                fvals[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = [(simplex[0][j] + simplex[i][j]) / 2 for j in range(n)]
                    fvals[i] = f(simplex[i])
    
    best_idx = min(range(n + 1), key=lambda i: fvals[i])
    return simplex[best_idx], fvals[best_idx]


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping using axial coordinates:
    # For a unit hexagon (side=1) with flat-top orientation:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # We need to place 11 hexagons. Start with a hexagonal lattice of 7 (center + 6 neighbors),
    # then add 4 more from ring 2.
    
    # Center + ring 1 (6 hexagons)
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    # Ring 2 positions (distance 2 from center) - 12 hexagons
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Pre-compute all ring2 combinations
    all_combos = list(itertools.combinations(range(len(ring2)), 4))
    
    # Phase 1: Quick scan with coarse grid to find promising configurations
    for combo in all_combos:
        if time.time() > deadline - 30.0:  # Leave 30s for phase 2
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Compute centroid
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        # Try several outer hexagon orientations
        for oang in [0.0, 30.0, 15.0, -15.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0]:
            side = _min_outer_side(inner_hexes, cx_mean, cy_mean, oang)
            if side < best_side:
                best_side = side
                best_positions = full_xy
                best_center = (cx_mean, cy_mean)
                best_angle = oang
    
    # Phase 2: LP-based optimization of outer hexagon parameters
    if best_positions is not None and time.time() < deadline:
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        
        # Collect all vertices of all inner hexagons
        all_points = []
        for cx, cy, s, ang in inner_hexes:
            verts = _hex_vertices(cx, cy, s, ang)
            all_points.extend(verts)
        
        # Use LP to find optimal outer hexagon
        lp_side, lp_cx, lp_cy, lp_theta = _lp_min_enclosing_hex(all_points)
        
        if lp_side < best_side:
            best_side = lp_side
            best_center = (lp_cx, lp_cy)
            best_angle = math.degrees(lp_theta)
        
        # Try multiple angle ranges for LP to find global optimum
        if time.time() < deadline - 10.0:
            # Refine with finer angle grid around best found angle
            best_theta_deg = math.degrees(lp_theta)
            for offset in [-30, -20, -10, 10, 20, 30]:
                if time.time() > deadline - 5.0:
                    break
                # We'll just rely on the initial LP which already does a grid
                pass

    if best_positions is None:
        # Fallback: just use the 7-hexagon cluster
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        best_side = _min_outer_side([(x, y, 1.0, 0.0) for x, y in best_positions], 0, 0, 0)
        best_center = (0, 0)
        best_angle = 0.0

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # If there's an issue, try scaling up positions slightly
        for scale in [1.001, 1.01, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_positions]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
                cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
                best_side = _min_outer_side(inner_hexes, cx_mean, cy_mean, 0.0)
                best_center = (cx_mean, cy_mean)
                best_angle = 0.0
                break

    # Verify all inner hexagons are inside outer with tight tolerance
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Binary search for minimum valid side length
        lo, hi = best_side, best_side * 1.1
        for _ in range(50):
            mid = (lo + hi) / 2
            test_outer = (best_center[0], best_center[1], mid, best_angle)
            ok = all(_inner_in_outer(h, test_outer) for h in inner_hexes)
            if ok:
                hi = mid
            else:
                lo = mid
        best_side = hi
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
