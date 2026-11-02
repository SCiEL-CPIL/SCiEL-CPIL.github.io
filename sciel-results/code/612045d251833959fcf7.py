"""Direct hexagon packing: 11 unit hexagons in a hexagonal lattice pattern with optimized outer hexagon."""
import math
import time
import itertools
import numpy as np
from scipy.optimize import linprog, minimize

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


def _min_outer_side_lp(hexes, oangle=0.0):
    """Use Linear Programming to find minimum side length and optimal center of outer hexagon
    with given rotation angle that contains all hexagons.
    
    Variables: [cx, cy, t] where t = side * sqrt(3)/2 (the apothem)
    Constraints: for each vertex (vx, vy) and each of 6 normal directions (nx, ny):
        (vx - cx)*cos(a) - (vy - cy)*sin(a) * nx + (vx - cx)*sin(a) + (vy - cy)*cos(a) * ny <= t
    which simplifies to:
        -vx*nx*cos(a) + vx*ny*sin(a) + cy*nx*sin(a) + cy*ny*cos(a) + cx*(-nx*cos(a) + ny*sin(a)) <= t - (vx*nx*cos(a) - vx*ny*sin(a))
    
    Let's define: for each vertex v=(vx,vy) and direction d=(nx,ny):
        proj = ((vx-cx)*cos(a) - (vy-cy)*sin(a))*nx + ((vx-cx)*sin(a) + (vy-cy)*cos(a))*ny
             = vx*(nx*cos(a) + ny*sin(a)) + vy*(ny*cos(a) - nx*sin(a)) 
               - cx*(nx*cos(a) + ny*sin(a)) - cy*(ny*cos(a) - nx*sin(a))
        Constraint: proj <= t
        => -cx*(nx*cos(a) + ny*sin(a)) - cy*(ny*cos(a) - nx*sin(a)) - t <= -(vx*(nx*cos(a) + ny*sin(a)) + vy*(ny*cos(a) - nx*sin(a)))
    """
    sqrt3 = math.sqrt(3)
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    
    # Collect all vertices
    all_verts = []
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.extend(verts)
    
    # 6 normal directions
    dir_vectors = []
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        dir_vectors.append((math.cos(theta), math.sin(theta)))
    
    n_constraints = len(all_verts) * 6
    n_vars = 3  # cx, cy, t
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    idx = 0
    for vx, vy in all_verts:
        for nx, ny in dir_vectors:
            # Coefficients for cx, cy, t
            # -cx*(nx*cos_a + ny*sin_a) - cy*(ny*cos_a - nx*sin_a) - t <= -(vx*(nx*cos_a + ny*sin_a) + vy*(ny*cos_a - nx*sin_a))
            coeff_cx = -(nx * cos_a + ny * sin_a)
            coeff_cy = -(ny * cos_a - nx * sin_a)
            coeff_t = -1.0
            rhs = -(vx * (nx * cos_a + ny * sin_a) + vy * (ny * cos_a - nx * sin_a))
            
            A_ub[idx, 0] = coeff_cx
            A_ub[idx, 1] = coeff_cy
            A_ub[idx, 2] = coeff_t
            b_ub[idx] = rhs
            idx += 1
    
    # Objective: minimize t
    c_obj = np.array([0.0, 0.0, 1.0])
    
    # Bounds: cx, cy unbounded; t >= 0
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            t_val = result.x[2]
            side = t_val * 2.0 / sqrt3
            return side, result.x[0], result.x[1]
        else:
            # Fallback to grid-based approach
            return _min_outer_side(hexes, 0.0, 0.0, oangle), 0.0, 0.0
    except Exception:
        return _min_outer_side(hexes, 0.0, 0.0, oangle), 0.0, 0.0


def _optimize_outer_hex(inner_hexes, initial_cx, initial_cy, initial_angle, max_iters=80, tol=1e-10):
    """Optimize the outer hexagon (center, angle) to minimize its side length.
    
    Uses LP for exact center+size given angle, then scipy.optimize for angle,
    with final high-precision polish.
    """
    sqrt3 = math.sqrt(3)
    
    # Evaluate using LP for exact solution given angle
    def eval_with_lp(angle):
        """Return (side, cx, cy) using LP for given angle."""
        side, cx, cy = _min_outer_side_lp(inner_hexes, angle)
        return side, cx, cy
    
    # Evaluate exact side for given (cx, cy, angle) without LP
    all_verts = []
    for cx, cy, s, ang in inner_hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.extend(verts)
    
    dir_vectors = []
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        dir_vectors.append((math.cos(theta), math.sin(theta)))
    
    def eval_side_exact(cx, cy, angle):
        """Exact evaluation of outer hexagon side length."""
        a = -math.radians(angle)
        cos_a = math.cos(a)
        sin_a = math.sin(a)
        max_val = 0.0
        for vx, vy in all_verts:
            dx = vx - cx
            dy = vy - cy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for nx, ny in dir_vectors:
                proj = lx * nx + ly * ny
                if proj > max_val:
                    max_val = proj
        return max_val * 2.0 / sqrt3
    
    # Get initial LP solution
    best_side, best_cx, best_cy = eval_with_lp(initial_angle)
    best_angle = initial_angle
    
    # Phase 1: Optimize angle using scipy with LP inner solver
    def objective(angle_arr):
        angle = angle_arr[0]
        side, _, _ = eval_with_lp(angle)
        return side
    
    # Try multiple starting angles
    start_angles = [initial_angle, initial_angle + 5, initial_angle - 5, 
                    initial_angle + 10, initial_angle - 10,
                    initial_angle + 15, initial_angle - 15,
                    initial_angle + 30, initial_angle - 30]
    
    for sa in start_angles:
        try:
            result = minimize(objective, [sa], method='Nelder-Mead',
                            options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 200, 'maxfev': 300})
            if result.fun < best_side - 1e-12:
                best_angle = result.x[0]
                best_side, best_cx, best_cy = eval_with_lp(best_angle)
        except Exception:
            pass
    
    # Phase 2: Fine-tune angle with small perturbations + LP
    step_a = 0.1  # degrees
    for iteration in range(30):
        improved = False
        for da in [step_a, -step_a, step_a*0.5, -step_a*0.5, step_a*0.25, -step_a*0.25,
                   step_a*0.1, -step_a*0.1]:
            new_angle = best_angle + da
            new_side, new_cx, new_cy = eval_with_lp(new_angle)
            if new_side < best_side - 1e-12:
                best_angle = new_angle
                best_side = new_side
                best_cx = new_cx
                best_cy = new_cy
                improved = True
        if not improved:
            step_a *= 0.5
            if step_a < 1e-10:
                break
    
    # Phase 3: Final high-precision polish on center (angle fixed)
    step_c = 1e-5
    for iteration in range(30):
        improved = False
        for dcx, dcy in [(step_c, 0), (-step_c, 0), (0, step_c), (0, -step_c),
                         (step_c*0.707, step_c*0.707), (-step_c*0.707, step_c*0.707),
                         (step_c*0.707, -step_c*0.707), (-step_c*0.707, -step_c*0.707)]:
            new_side = eval_side_exact(best_cx + dcx, best_cy + dcy, best_angle)
            if new_side < best_side - 1e-12:
                best_cx += dcx
                best_cy += dcy
                best_side = new_side
                improved = True
        if not improved:
            step_c *= 0.5
            if step_c < 1e-12:
                break
    
    return best_cx, best_cy, best_angle, best_side


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
    
    # Ring 2 positions (distance 2 from center) - 14 hexagons
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Try all combinations of 4 from ring2
    # C(14, 4) = 1001 combinations
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Compute centroid as initial guess
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        # Try several initial outer hexagon configurations
        initial_candidates = [
            (cx_mean, cy_mean, 0.0),
            (cx_mean, cy_mean, 30.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 30.0),
            (cx_mean, cy_mean, 15.0),
            (cx_mean, cy_mean, -15.0),
        ]
        
        for ocx, ocy, oang in initial_candidates:
            # First evaluate quickly using LP
            side_quick, _, _ = _min_outer_side_lp(inner_hexes, oang)
            
            # If this looks promising, optimize
            if side_quick < best_side * 1.01:
                opt_cx, opt_cy, opt_angle, opt_side = _optimize_outer_hex(
                    inner_hexes, ocx, ocy, oang, max_iters=80
                )
                
                if opt_side < best_side:
                    best_side = opt_side
                    best_positions = full_xy
                    best_center = (opt_cx, opt_cy)
                    best_angle = opt_angle

    if best_positions is None:
        # Fallback: just use the 7-hexagon cluster
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        cx_mean = sum(x for x, y in best_positions) / 7.0
        cy_mean = sum(y for x, y in best_positions) / 7.0
        best_cx, best_cy, best_angle, best_side = _optimize_outer_hex(
            inner_hexes, cx_mean, cy_mean, 0.0, max_iters=80
        )
        best_center = (best_cx, best_cy)
    
    # Also try optimizing with non-zero inner hexagon rotations for the best configuration
    if best_positions is not None and time.time() < deadline - 30:
        # Try rotating all inner hexagons by the same angle
        for rot_angle in [0.0, 15.0, 30.0, 45.0, 60.0]:
            if time.time() > deadline - 10:
                break
            rotated_hexes = [(x, y, 1.0, rot_angle) for x, y in best_positions]
            cx_mean = sum(x for x, y, s, a in rotated_hexes) / len(rotated_hexes)
            cy_mean = sum(y for x, y, s, a in rotated_hexes) / len(rotated_hexes)
            
            for oang in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]:
                if time.time() > deadline - 10:
                    break
                opt_cx, opt_cy, opt_angle, opt_side = _optimize_outer_hex(
                    rotated_hexes, cx_mean, cy_mean, oang, max_iters=50
                )
                if opt_side < best_side - 1e-10:
                    best_side = opt_side
                    best_center = (opt_cx, opt_cy)
                    best_angle = opt_angle
                    # Store the rotated inner hexagons
                    best_positions = [(x, y) for x, y, s, a in rotated_hexes]
                    best_inner_angle = rot_angle
                    break

    # Build final inner hexagons
    if 'best_inner_angle' in locals():
        inner_hexes = [(x, y, 1.0, best_inner_angle) for x, y in best_positions]
    else:
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
                best_cx, best_cy, best_angle, best_side = _optimize_outer_hex(
                    inner_hexes, cx_mean, cy_mean, 0.0, max_iters=80
                )
                best_center = (best_cx, best_cy)
                break

    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Increase side slightly to ensure containment
        best_side *= 1.001
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    
    # Final verification and safety margin
    if time.time() < deadline - 5:
        # Double-check with a slightly larger tolerance
        for h in inner_hexes:
            if not _inner_in_outer(h, (best_center[0], best_center[1], best_side * 1.0001, best_angle)):
                best_side *= 1.0005
                break

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
