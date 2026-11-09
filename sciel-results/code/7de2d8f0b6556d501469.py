"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)
    rng = random.Random(42)

    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon in CCW order."""
        vertices = []
        angle_rad = math.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def check_overlap(centers, angles, eps=1e-9):
        """Check if any two inner hexagons overlap using polygon intersection."""
        for i in range(N):
            vi = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            for j in range(i + 1, N):
                vj = get_hex_vertices(centers[j][0], centers[j][1], angles[j])
                if _polygons_intersect(vi, vj):
                    return False
        return True

    def _polygons_intersect(poly1, poly2):
        """Check if two convex polygons intersect (including one containing the other)."""
        # Check if any vertex of poly1 is inside poly2
        for p in poly1:
            if _point_in_convex(p[0], p[1], poly2):
                return True
        # Check if any vertex of poly2 is inside poly1
        for p in poly2:
            if _point_in_convex(p[0], p[1], poly1):
                return True
        # Check for edge intersections
        for i in range(6):
            p1 = poly1[i]
            p2 = poly1[(i + 1) % 6]
            for j in range(6):
                p3 = poly2[j]
                p4 = poly2[(j + 1) % 6]
                if _segments_intersect(p1, p2, p3, p4):
                    return True
        return False

    def _point_in_convex(px, py, poly):
        """Check if point is inside convex polygon (CCW order)."""
        for i in range(6):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % 6]
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if cross < 0:
                return False
        return True

    def _segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        c1 = cross(p3, p4, p1)
        c2 = cross(p3, p4, p2)
        c3 = cross(p1, p2, p3)
        c4 = cross(p1, p2, p4)
        
        if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
           ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
            return True
        
        # Collinear cases (rare in practice, but handle for robustness)
        if abs(c1) < 1e-12 and _on_segment(p3, p1, p4):
            return True
        if abs(c2) < 1e-12 and _on_segment(p3, p2, p4):
            return True
        if abs(c3) < 1e-12 and _on_segment(p1, p3, p2):
            return True
        if abs(c4) < 1e-12 and _on_segment(p1, p4, p2):
            return True
        
        return False

    def _on_segment(p, q, r):
        """Check if point q lies on segment pr."""
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and \
           min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    def enclosing_side_length(centers, angles, outer_angle=0.0):
        """Compute minimal side length of outer hexagon containing all inner hexagons."""
        # 6 outward normal directions for outer hexagon
        normals = []
        for k in range(6):
            angle = math.radians(outer_angle + 30 + k * 60)
            normals.append((math.cos(angle), math.sin(angle)))
        
        max_extent = 0.0
        for i in range(N):
            cx, cy = centers[i]
            inner_angle_rad = math.radians(angles[i])
            
            for nk in range(6):
                nx, ny = normals[nk]
                n_angle = math.atan2(ny, nx)
                
                max_dot = -2.0
                for j in range(6):
                    v_angle = inner_angle_rad + j * math.pi / 3.0
                    dot_val = math.cos(v_angle - n_angle)
                    if dot_val > max_dot:
                        max_dot = dot_val
                
                extent = cx * nx + cy * ny + max_dot
                if extent > max_extent:
                    max_extent = extent
        
        # max_extent is the apothem; side = 2 * apothem / sqrt(3)
        return 2.0 * max_extent / sqrt3

    # Initial hexagonal lattice configuration
    def grid_to_cart(q, r):
        x = sqrt3 * q + sqrt3 / 2 * r
        y = 1.5 * r
        return (x, y)

    init_grid_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]

    centers = [grid_to_cart(q, r) for q, r in init_grid_coords]
    angles = [0.0] * N

    if not check_overlap(centers, angles):
        centers = [(5.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        angles = [0.0] * N

    best_L = enclosing_side_length(centers, angles, 0.0)
    best_centers = [c[:] for c in centers]
    best_angles = angles[:]
    best_outer_angle = 0.0

    # Objective function: minimize outer side length with overlap penalty
    def objective(params):
        if time.time() > deadline:
            return 1e10
        centers_opt = []
        angles_opt = []
        for i in range(N):
            centers_opt.append((params[i * 3], params[i * 3 + 1]))
            angles_opt.append(params[i * 3 + 2])
        outer_angle = params[36]
        
        if not check_overlap(centers_opt, angles_opt):
            # Penalty: return large value plus actual L to guide away
            L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
            # Estimate violation magnitude for smoother penalty
            violation = 0.0
            for i in range(N):
                vi = get_hex_vertices(centers_opt[i][0], centers_opt[i][1], angles_opt[i])
                for j in range(i + 1, N):
                    vj = get_hex_vertices(centers_opt[j][0], centers_opt[j][1], angles_opt[j])
                    for p in vi:
                        if _point_in_convex(p[0], p[1], vj):
                            violation += 1.0
                    for p in vj:
                        if _point_in_convex(p[0], p[1], vi):
                            violation += 1.0
            return 1000.0 + L + 10.0 * violation
        
        return enclosing_side_length(centers_opt, angles_opt, outer_angle)

    init_params = []
    for i in range(N):
        init_params.append(centers[i][0])
        init_params.append(centers[i][1])
        init_params.append(angles[i])
    init_params.append(0.0)

    # Multi-start local search with Nelder-Mead
    n_starts = 8
    for s in range(n_starts):
        if time.time() > deadline - 15.0:
            break
        
        # Perturb initial configuration for diversity
        if s == 0:
            current_params = init_params[:]
        else:
            current_params = init_params[:]
            for i in range(N):
                current_params[i * 3] += rng.uniform(-1.0, 1.0)
                current_params[i * 3 + 1] += rng.uniform(-1.0, 1.0)
                current_params[i * 3 + 2] += rng.uniform(-30.0, 30.0)
            current_params[36] = rng.uniform(-30.0, 30.0)
        
        try:
            res = minimize(
                objective,
                current_params,
                method='Nelder-Mead',
                options={'maxiter': 1500, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
            )
            
            if res.fun < objective(current_params) or res.success:
                current_params = res.x
                centers_opt = []
                angles_opt = []
                for i in range(N):
                    centers_opt.append((current_params[i * 3], current_params[i * 3 + 1]))
                    angles_opt.append(current_params[i * 3 + 2])
                outer_angle = current_params[36]
                
                if check_overlap(centers_opt, angles_opt):
                    L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
                    if L < best_L:
                        best_L = L
                        best_centers = centers_opt
                        best_angles = angles_opt
                        best_outer_angle = outer_angle
        except Exception:
            continue

    # Final validation
    if not check_overlap(best_centers, best_angles):
        best_centers = [grid_to_cart(q, r) for q, r in init_grid_coords]
        best_angles = [0.0] * N
        best_L = enclosing_side_length(best_centers, best_angles, 0.0)
        best_outer_angle = 0.0

    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], best_angles[i]])

    center = (0.0, 0.0)
    side = best_L
    angle = best_outer_angle

    return inner, center, side, angle
# EVOLVE-BLOCK-END
