"""Direct hexagon packing: 11 unit hexagons in a hexagonal lattice pattern.
Uses Nelder-Mead optimization for the outer hexagon parameters."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import minimize

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
    that contains all hexagons in `hexes`. Vectorized with NumPy."""
    if not hexes:
        return 0.0
    
    # Precompute all vertices for all hexagons
    all_verts = []
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.extend(verts)
    
    if not all_verts:
        return 0.0
        
    verts_arr = np.array(all_verts)
    # Translate to outer center
    verts_arr[:, 0] -= ox
    verts_arr[:, 1] -= oy
    
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    
    # Rotate vertices
    lx = verts_arr[:, 0] * cos_a - verts_arr[:, 1] * sin_a
    ly = verts_arr[:, 0] * sin_a + verts_arr[:, 1] * cos_a
    
    # Projection directions for regular hexagon (apothem directions)
    # Normals are at 30, 90, 150, 210, 270, 330 degrees
    angles = np.radians([30, 90, 150, 210, 270, 330])
    nx = np.cos(angles)
    ny = np.sin(angles)
    
    # Compute projections: (N_vertices, 6)
    # proj[i, k] = lx[i] * nx[k] + ly[i] * ny[k]
    # Using broadcasting: lx (N, 1) * nx (1, 6) + ly (N, 1) * ny (1, 6)
    projs = lx[:, np.newaxis] * nx[np.newaxis, :] + ly[:, np.newaxis] * ny[np.newaxis, :]
    
    # Max projection for each direction
    max_projs = np.max(projs, axis=0)
    
    # The side length S is related to the apothem A by A = S * sqrt(3) / 2
    # So S = A * 2 / sqrt(3)
    # The maximum apothem required is max(max_projs)
    max_apothem = np.max(max_projs)
    
    return max_apothem * 2.0 / math.sqrt(3)


def _min_outer_side_smooth(hexes, ox, oy, oangle=0.0, beta=50.0):
    """Smooth approximation of _min_outer_side using soft-max (log-sum-exp).
    This provides a differentiable surrogate for gradient-based optimization."""
    if not hexes:
        return 0.0
    
    # Precompute all vertices for all hexagons
    all_verts = []
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.extend(verts)
    
    if not all_verts:
        return 0.0
        
    verts_arr = np.array(all_verts)
    # Translate to outer center
    verts_arr[:, 0] -= ox
    verts_arr[:, 1] -= oy
    
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    
    # Rotate vertices
    lx = verts_arr[:, 0] * cos_a - verts_arr[:, 1] * sin_a
    ly = verts_arr[:, 0] * sin_a + verts_arr[:, 1] * cos_a
    
    # Projection directions for regular hexagon (apothem directions)
    angles = np.radians([30, 90, 150, 210, 270, 330])
    nx = np.cos(angles)
    ny = np.sin(angles)
    
    # Compute projections: (N_vertices, 6)
    projs = lx[:, np.newaxis] * nx[np.newaxis, :] + ly[:, np.newaxis] * ny[np.newaxis, :]
    
    # Soft-max for each direction: (6,)
    # For numerical stability, subtract max before exp
    max_per_dir = np.max(projs, axis=0)
    exp_projs = np.exp(beta * (projs - max_per_dir[np.newaxis, :]))
    soft_max = max_per_dir + np.log(np.sum(exp_projs, axis=0)) / beta
    
    # Soft-max over the 6 directions
    max_soft = np.max(soft_max)
    exp_soft = np.exp(beta * (soft_max - max_soft))
    soft_max_all = max_soft + np.log(np.sum(exp_soft)) / beta
    
    return soft_max_all * 2.0 / math.sqrt(3)


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
    
    # Phase 1: Scan all configurations
    # With vectorized _min_outer_side, we can afford a finer angle grid
    # and evaluate all 495 combos quickly.
    for combo in all_combos:
        if time.time() > deadline - 60.0:  # Leave 60s for phase 2
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Compute centroid
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        # Try several outer hexagon orientations (finer grid: 1.0 degrees)
        for oang in np.arange(0.0, 180.0, 1.0):
            side = _min_outer_side(inner_hexes, cx_mean, cy_mean, oang)
            if side < best_side:
                best_side = side
                best_positions = full_xy
                best_center = (cx_mean, cy_mean)
                best_angle = oang
    
    # Phase 2: Nelder-Mead optimization of outer hexagon parameters
    if best_positions is not None and time.time() < deadline:
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        
        def objective(params):
            cx, cy, ang = params
            return _min_outer_side(inner_hexes, cx, cy, ang)
        
        # Run Nelder-Mead from the best point found
        x0 = [best_center[0], best_center[1], best_angle]
        bounds = [(-5, 5), (-5, 5), (-90, 90)]
        
        x_best, f_best = _nelder_mead(objective, x0, bounds, max_iter=500, tol=1e-10)
        
        if f_best < best_side:
            best_side = f_best
            best_center = (x_best[0], x_best[1])
            best_angle = x_best[2]
        
        # Try multiple starting points for Nelder-Mead to avoid local minima
        if time.time() < deadline - 10.0:
            for start_ang in [0.0, 30.0, -30.0, 15.0, -15.0, 45.0, -45.0]:
                if time.time() > deadline - 5.0:
                    break
                cx_mean = sum(x for x, y in best_positions) / 11.0
                cy_mean = sum(y for x, y in best_positions) / 11.0
                x0 = [cx_mean, cy_mean, start_ang]
                x_best, f_best = _nelder_mead(objective, x0, bounds, max_iter=300, tol=1e-10)
                if f_best < best_side:
                    best_side = f_best
                    best_center = (x_best[0], x_best[1])
                    best_angle = x_best[2]

    if best_positions is None:
        # Fallback: just use the 7-hexagon cluster
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        best_side = _min_outer_side([(x, y, 1.0, 0.0) for x, y in best_positions], 0, 0, 0)
        best_center = (0, 0)
        best_angle = 0.0

    # Final polish: high-precision optimization of outer hexagon parameters
    if best_positions is not None and time.time() < deadline:
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        
        def objective(params):
            cx, cy, ang = params
            return _min_outer_side(inner_hexes, cx, cy, ang)
        
        # Use scipy.optimize.minimize with Nelder-Mead for robustness
        # and L-BFGS-B for speed if possible, but Nelder-Mead handles non-smoothness better.
        # We'll run multiple restarts from the best point and nearby points.
        
        start_points = [
            [best_center[0], best_center[1], best_angle],
            [best_center[0], best_center[1], best_angle + 1.0],
            [best_center[0], best_center[1], best_angle - 1.0],
            [best_center[0] + 0.1, best_center[1], best_angle],
            [best_center[0] - 0.1, best_center[1], best_angle],
            [best_center[0], best_center[1] + 0.1, best_angle],
            [best_center[0], best_center[1] - 0.1, best_angle],
        ]
        
        for x0 in start_points:
            if time.time() > deadline - 5.0:
                break
            try:
                res = minimize(objective, x0, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
                if res.fun < best_side:
                    best_side = res.fun
                    best_center = (res.x[0], res.x[1])
                    best_angle = res.x[2]
            except Exception:
                pass
        
        # Final L-BFGS-B polish if it seems smooth enough (it might not be, but worth a try)
        if time.time() < deadline - 2.0:
            try:
                res = minimize(objective, [best_center[0], best_center[1], best_angle], method='L-BFGS-B', 
                              bounds=[(-5, 5), (-5, 5), (-90, 90)],
                              options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-14})
                if res.fun < best_side:
                    best_side = res.fun
                    best_center = (res.x[0], res.x[1])
                    best_angle = res.x[2]
            except Exception:
                pass

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

    # Verify disjointness with a small gap to ensure strict separation
    if not _all_disjoint(inner_hexes):
        # If there's an issue, try scaling up positions slightly with a larger gap
        for scale in [1.002, 1.005, 1.01, 1.02, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_positions]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
                cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
                best_side = _min_outer_side(inner_hexes, cx_mean, cy_mean, best_angle)
                best_center = (cx_mean, cy_mean)
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
