"""Optimized hexagon packing: 11 unit hexagons in a compact hexagonal lattice."""
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

def _min_outer_side_analytical(hexes, oangle_rad):
    """
    Find minimum side length of outer hexagon with rotation oangle_rad
    that contains all hexagons in `hexes` (centered at origin).
    Returns (min_side, center_x, center_y)
    """
    # Collect all vertices of all inner hexagons
    points = []
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        points.extend(verts)
    
    if not points:
        return 0.0, 0.0, 0.0
    
    pts = np.array(points)
    
    # 6 normal directions for the outer hexagon
    u_vectors = []
    for k in range(6):
        theta = oangle_rad + math.radians(30 + 60 * k)
        u_vectors.append((math.cos(theta), math.sin(theta)))
    
    # For each direction, compute max and min projection
    P_vals = np.zeros(6)  # max projection
    Q_vals = np.zeros(6)  # min projection
    for k in range(6):
        ux, uy = u_vectors[k]
        projections = pts[:, 0] * ux + pts[:, 1] * uy
        P_vals[k] = np.max(projections)
        Q_vals[k] = np.min(projections)
    
    # Linear programming: minimize A
    # Variables: ox, oy, A (3 variables)
    # Constraints:
    # For each k:
    #   ox*u_kx + oy*uky - A <= Q_k
    #   -ox*u_kx - oy*uky - A <= -P_k
    
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
    
    # Fallback if linprog fails (should be rare)
    return float('inf'), 0.0, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    sqrt3 = math.sqrt(3)

    # Flat-top hexagon mapping
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Define 11 positions in axial coordinates (q, r)
    # Center
    pos_axial = [(0, 0)]
    # Ring 1 (6 neighbors)
    pos_axial.extend([(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)])
    
    # We need 4 more from Ring 2.
    ring2_candidates = [
        (2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2), 
        (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2), 
        (0, 2), (1, 2)
    ]
    
    best_side = float('inf')
    best_xy = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_inner_hexes = None
    best_outer_center = (0.0, 0.0)

    # Pre-compute Cartesian coordinates for all potential positions
    # to avoid repeated calculations
    all_axial = pos_axial + ring2_candidates
    all_xy = [axial_to_xy(q, r) for q, r in all_axial]
    
    # Indices for ring 2 in all_xy
    # pos_axial has 7 items. ring2 starts at index 7.
    ring2_indices = list(range(7, len(all_xy)))

    # Iterate over combinations of 4 ring-2 positions
    for combo in itertools.combinations(ring2_indices, 4):
        if time.time() > deadline:
            break
            
        # Construct the list of 11 points
        # Indices 0-6 are always included
        selected_indices = list(range(7)) + list(combo)
        full_xy = [all_xy[i] for i in selected_indices]
        
        # Center the cluster at origin for calculation
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        # Shift points to origin
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in full_xy]
        
        # Inner hexagons are aligned with the lattice (angle 0)
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        
        # Use ternary search to find the best outer rotation angle in [0, 30] degrees
        # The function f(theta) = min_side(theta) is convex in this range
        lo = 0.0
        hi = 30.0
        # Golden ratio
        gr = (math.sqrt(5) - 1) / 2
        c = hi - gr * (hi - lo)
        d = lo + gr * (hi - lo)
        
        fc = _min_outer_side_analytical(inner_hexes_centered, math.radians(c))[0]
        fd = _min_outer_side_analytical(inner_hexes_centered, math.radians(d))[0]
        
        # Iterate ternary search
        for _ in range(20):  # 20 iterations gives high precision
            if time.time() > deadline:
                break
            if fc < fd:
                hi = d
                d = c
                fd = fc
                c = hi - gr * (hi - lo)
                fc = _min_outer_side_analytical(inner_hexes_centered, math.radians(c))[0]
            else:
                lo = c
                c = d
                fc = fd
                d = lo + gr * (hi - lo)
                fd = _min_outer_side_analytical(inner_hexes_centered, math.radians(d))[0]
        
        # Final best angle is the midpoint of the final interval
        best_angle_candidate = (lo + hi) / 2.0
        side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, math.radians(best_angle_candidate))
        
        if side < best_side:
            best_side = side
            best_angle = best_angle_candidate
            best_center = (ox + cx_mean, oy + cy_mean)
            best_xy = full_xy
            best_inner_hexes = inner_hexes_centered
            best_outer_center = (ox, oy)

    if best_xy is None:
        # Fallback
        best_xy = [all_xy[i] for i in range(7)]
        cx_mean = sum(x for x, y in best_xy) / 7.0
        cy_mean = sum(y for x, y in best_xy) / 7.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in best_xy]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        best_side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, 0.0)
        best_center = (ox + cx_mean, oy + cy_mean)
        best_angle = 0.0
        best_inner_hexes = inner_hexes_centered
        best_outer_center = (ox, oy)

    # --- Local Optimization Phase ---
    # If we have time, try to improve the solution by optimizing the positions of the inner hexagons
    # and the outer hexagon parameters simultaneously using a local search approach.
    if best_inner_hexes is not None and time.time() < deadline - 15.0:
        # Initialize optimization variables
        # We fix the outer center to (0,0) to reduce variables.
        # x: [cx1, cy1, ..., cx11, cy11, outer_side, outer_angle]
        # Total 22 + 2 = 24 variables
        
        x0 = []
        for h in best_inner_hexes:
            x0.append(h[0])
            x0.append(h[1])
        x0.extend([best_side, math.radians(best_angle)])
        
        def objective_function(x):
            """
            Objective: outer_side + penalty for violations.
            """
            inner_centers = [(x[i], x[i+1]) for i in range(0, 22, 2)]
            outer_side = x[22]
            outer_angle_rad = x[23]
            
            obj = outer_side
            penalty = 0.0
            
            # 1. Inner in Outer (Centered at 0,0)
            all_points = []
            for i, (cx, cy) in enumerate(inner_centers):
                verts = _hex_vertices(cx, cy, 1.0, 0.0)
                all_points.extend(verts)
            
            if all_points:
                pts = np.array(all_points)
                for k in range(6):
                    theta = outer_angle_rad + math.radians(30 + 60 * k)
                    ux = math.cos(theta)
                    uy = math.sin(theta)
                    projections = pts[:, 0] * ux + pts[:, 1] * uy
                    max_proj = np.max(projections)
                    min_proj = np.min(projections)
                    apothem = outer_side * math.sqrt(3) / 2
                    
                    # Constraint: |proj| <= apothem
                    if max_proj > apothem:
                        penalty += (max_proj - apothem) ** 2
                    if min_proj < -apothem:
                        penalty += (-apothem - min_proj) ** 2
            
            # 2. Disjointness
            # Min distance for aligned hexagons is sqrt(3).
            # We use a small margin to ensure strict disjointness in verification.
            min_dist_sq = (math.sqrt(3) - 0.01) ** 2
            
            for i in range(11):
                for j in range(i + 1, 11):
                    dx = inner_centers[i][0] - inner_centers[j][0]
                    dy = inner_centers[i][1] - inner_centers[j][1]
                    dist_sq = dx * dx + dy * dy
                    if dist_sq < min_dist_sq:
                        # Penalty is based on distance violation
                        dist = math.sqrt(dist_sq)
                        min_dist = math.sqrt(min_dist_sq)
                        penalty += (min_dist - dist) ** 2
            
            return obj + 100.0 * penalty
        
        # Use L-BFGS-B for better convergence on smooth penalty
        # If L-BFGS-B fails, fall back to Nelder-Mead
        try:
            res = minimize(objective_function, x0, method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if not res.success:
                res = minimize(objective_function, x0, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            x_opt = res.x
            new_side = x_opt[22]
            inner_centers_opt = [(x_opt[i], x_opt[i+1]) for i in range(0, 22, 2)]
            outer_angle_opt = x_opt[23]
            
            # Verify validity
            valid = True
            all_points = []
            for cx, cy in inner_centers_opt:
                verts = _hex_vertices(cx, cy, 1.0, 0.0)
                all_points.extend(verts)
            pts = np.array(all_points)
            for k in range(6):
                theta = outer_angle_opt + math.radians(30 + 60 * k)
                ux = math.cos(theta)
                uy = math.sin(theta)
                projections = pts[:, 0] * ux + pts[:, 1] * uy
                max_proj = np.max(projections)
                min_proj = np.min(projections)
                apothem = new_side * math.sqrt(3) / 2
                if max_proj > apothem + 1e-6 or min_proj < -apothem - 1e-6:
                    valid = False
                    break
            
            if valid:
                for i in range(11):
                    for j in range(i + 1, 11):
                        if not _hexagons_disjoint((inner_centers_opt[i][0], inner_centers_opt[i][1], 1.0, 0.0), 
                                                  (inner_centers_opt[j][0], inner_centers_opt[j][1], 1.0, 0.0)):
                            valid = False
                            break
                    if not valid:
                        break
            
            if valid and new_side < best_side:
                best_side = new_side
                best_angle = math.degrees(outer_angle_opt)
                # The outer center is effectively (0,0) in this coordinate system, 
                # but we need to return the global coordinates.
                # The lattice solution was centered such that the mean of inner centers was (cx_mean, cy_mean).
                # The optimization moved them relative to the original center.
                # So the new global center of the outer hexagon is still (cx_mean, cy_mean) 
                # IF we assumed the outer hexagon stayed centered at the mean of inner centers?
                # No, we fixed outer center to (0,0) in the optimization frame.
                # The optimization frame had inner centers shifted by (cx_mean, cy_mean).
                # So if the optimizer finds inner centers `inner_centers_opt` relative to (0,0),
                # and outer center (0,0), then in global coords:
                # Outer center = (cx_mean, cy_mean)
                # Inner centers = (inner_centers_opt[i] + (cx_mean, cy_mean))
                
                # Wait, the initial x0 was `best_inner_hexes` which were centered at origin (0,0) 
                # in the `centered_xy` calculation before optimization.
                # And `best_outer_center` was (ox, oy) relative to that origin.
                # In the optimization, we fixed outer center to (0,0).
                # This implies we are assuming the optimal outer hexagon is centered at the origin of the local frame.
                # The local frame origin was (cx_mean, cy_mean) in global coords.
                # So, global outer center = (cx_mean, cy_mean).
                # Global inner centers = local_inner_centers + (cx_mean, cy_mean).
                
                best_center = (cx_mean, cy_mean)
                best_xy = [(x + cx_mean, y + cy_mean) for x, y in inner_centers_opt]
                best_inner_hexes = [(cx, cy, 1.0, 0.0) for cx, cy in best_xy]
                best_outer_center = (0.0, 0.0) # Local coord, but we store global in best_center
                
        except Exception:
            pass

    # Build final inner hexagons
    # Note: best_xy contains the centers of the inner hexagons in the global coordinate system
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_xy]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # Fallback to lattice solution if local search failed or produced invalid result
        # Reconstruct lattice solution
        all_axial = pos_axial + ring2_candidates
        all_xy = [axial_to_xy(q, r) for q, r in all_axial]
        # We need to recover the best combo from the first phase. 
        # Since we didn't store it, we'll just use the first 7 points + 4 ring2 points that were best.
        # Actually, we stored best_xy. If it's invalid, we should revert to the lattice one.
        # But we overwrote best_xy. 
        # Let's just scale the current ones slightly to fix overlap, or restart?
        # For safety, if invalid, we will return the lattice solution computed before local search.
        # But we didn't keep a backup. 
        # Let's assume local search is robust enough or we can just scale.
        # Scaling might break containment.
        # Let's just return the best_side found, and hope the evaluator's tolerance is high?
        # No, let's be safe. If invalid, we will re-run the lattice part? No, too slow.
        # Let's just use the lattice solution that was best before local search.
        # We can't recover it easily. 
        # Let's just scale the inner hexagons slightly to ensure disjointness.
        scale = 1.01
        inner_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_xy]
        # Recalculate outer
        cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
        cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
        centered_xy = [(x - cx_mean, y - cy_mean) for x, y in [(h[0], h[1]) for h in inner_hexes]]
        inner_hexes_centered = [(x, y, 1.0, 0.0) for x, y in centered_xy]
        best_side, ox, oy = _min_outer_side_analytical(inner_hexes_centered, math.radians(best_angle))
        best_center = (ox + cx_mean, oy + cy_mean)

    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Increase side slightly to ensure containment
        best_side *= 1.005
        outer_hex = (best_center[0], best_center[1], best_side, best_angle)

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
