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
    deadline = start_time + 140.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)
    rng = random.Random(42)  # Fixed seed for reproducibility in evaluation, though randomness helps exploration

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

    def point_in_convex(px, py, poly):
        """Check if point is inside convex polygon (CCW order)."""
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if cross < -1e-10:
                return False
        return True

    def segments_intersect(p1, p2, p3, p4):
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
        return False

    def polygons_intersect(poly1, poly2):
        """Check if two convex polygons intersect."""
        # 1. Check if any vertex of poly1 is inside poly2
        for p in poly1:
            if point_in_convex(p[0], p[1], poly2):
                return True
        # 2. Check if any vertex of poly2 is inside poly1
        for p in poly2:
            if point_in_convex(p[0], p[1], poly1):
                return True
        # 3. Check for edge intersections
        for i in range(len(poly1)):
            p1 = poly1[i]
            p2 = poly1[(i + 1) % len(poly1)]
            for j in range(len(poly2)):
                p3 = poly2[j]
                p4 = poly2[(j + 1) % len(poly2)]
                if segments_intersect(p1, p2, p3, p4):
                    return True
        return False

    def check_overlap(centers, angles):
        """Check if any two inner hexagons overlap."""
        # Pre-compute vertices for efficiency
        polys = []
        for i in range(N):
            polys.append(get_hex_vertices(centers[i][0], centers[i][1], angles[i]))
        
        for i in range(N):
            for j in range(i + 1, N):
                if polygons_intersect(polys[i], polys[j]):
                    return False
        return True

    def enclosing_side_length(centers, angles, outer_angle=0.0):
        """
        Compute minimal side length of outer hexagon containing all inner hexagons.
        Uses projection method: L = 2 * max_extent / sqrt(3)
        """
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

                # Max dot product of any vertex of inner hex i with normal n_k
                max_dot = -2.0
                for j in range(6):
                    v_angle = inner_angle_rad + j * math.pi / 3.0
                    dot_val = math.cos(v_angle - n_angle)
                    if dot_val > max_dot:
                        max_dot = dot_val

                extent = cx * nx + cy * ny + max_dot
                if extent > max_extent:
                    max_extent = extent

        return 2.0 * max_extent / sqrt3

    def grid_to_cart(q, r):
        x = sqrt3 * q + sqrt3 / 2 * r
        y = 1.5 * r
        return (x, y)

    # Initial hexagonal lattice configuration
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

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        centers_opt = []
        angles_opt = []
        for i in range(N):
            centers_opt.append((params[i * 3], params[i * 3 + 1]))
            angles_opt.append(params[i * 3 + 2])
        outer_angle = params[36]

        # Calculate L first, it's needed for penalty scaling if we were doing constraint-based, 
        # but here we just minimize L + penalty.
        L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
        
        if not check_overlap(centers_opt, angles_opt):
            # Calculate overlap violation magnitude for smoother gradient
            # Simple count is fine for Nelder-Mead, but let's try to measure depth
            violation = 0.0
            polys = []
            for i in range(N):
                polys.append(get_hex_vertices(centers_opt[i][0], centers_opt[i][1], angles_opt[i]))
            for i in range(N):
                for j in range(i + 1, N):
                    # Check vertex in poly for violation count
                    for p in polys[i]:
                        if point_in_convex(p[0], p[1], polys[j]):
                            violation += 1.0
                    for p in polys[j]:
                        if point_in_convex(p[0], p[1], polys[i]):
                            violation += 1.0
            # Heavier penalty for overlap
            return 1000.0 + L + 10.0 * violation
        
        return L

    init_params = []
    for i in range(N):
        init_params.append(centers[i][0])
        init_params.append(centers[i][1])
        init_params.append(angles[i])
    init_params.append(0.0)

    # Multi-start with different strategies
    n_starts = 15
    for s in range(n_starts):
        if time.time() > deadline - 15.0:
            break

        if s == 0:
            current_params = init_params[:]
        elif s == 1:
            # Perturb with small random offsets
            current_params = init_params[:]
            for i in range(N):
                current_params[i * 3] += rng.uniform(-0.5, 0.5)
                current_params[i * 3 + 1] += rng.uniform(-0.5, 0.5)
                current_params[i * 3 + 2] += rng.uniform(-15.0, 15.0)
            current_params[36] = rng.uniform(-15.0, 15.0)
        elif s == 2:
            # Larger perturbation
            current_params = init_params[:]
            for i in range(N):
                current_params[i * 3] += rng.uniform(-1.5, 1.5)
                current_params[i * 3 + 1] += rng.uniform(-1.5, 1.5)
                current_params[i * 3 + 2] += rng.uniform(-45.0, 45.0)
            current_params[36] = rng.uniform(-45.0, 45.0)
        elif s == 3:
            # Try rotating all hexagons by 30 degrees
            current_params = init_params[:]
            for i in range(N):
                current_params[i * 3 + 2] = 30.0
            current_params[36] = 0.0
        elif s == 4:
            # Try rotating all hexagons by 15 degrees
            current_params = init_params[:]
            for i in range(N):
                current_params[i * 3 + 2] = 15.0
            current_params[36] = 15.0
        elif s == 5:
            # Random configuration within a bounding box
            current_params = []
            for i in range(N):
                current_params.append(rng.uniform(-4, 4))
                current_params.append(rng.uniform(-4, 4))
                current_params.append(rng.uniform(-60, 60))
            current_params.append(rng.uniform(-60, 60))
        elif s == 6:
            # Compact circular arrangement
            current_params = []
            for i in range(N):
                angle = 2 * math.pi * i / N
                r = 2.5
                current_params.append(r * math.cos(angle))
                current_params.append(r * math.sin(angle))
                current_params.append(rng.uniform(-10, 10))
            current_params.append(0.0)
        elif s == 7:
            # Two rings
            current_params = []
            for i in range(6):
                angle = 2 * math.pi * i / 6
                r = 2.0
                current_params.append(r * math.cos(angle))
                current_params.append(r * math.sin(angle))
                current_params.append(0.0)
            for i in range(6):
                angle = 2 * math.pi * (i + 0.5) / 6
                r = 4.0
                current_params.append(r * math.cos(angle))
                current_params.append(r * math.sin(angle))
                current_params.append(0.0)
            current_params.append(0.0)
        elif s == 8:
            # Hexagonal lattice but rotated 30 deg
            current_params = []
            for i, (q, r) in enumerate(init_grid_coords):
                cx, cy = grid_to_cart(q, r)
                # Rotate coords by 30 deg
                rad = math.radians(30)
                nx = cx * math.cos(rad) - cy * math.sin(rad)
                ny = cx * math.sin(rad) + cy * math.cos(rad)
                current_params.append(nx)
                current_params.append(ny)
                current_params.append(30.0)
            current_params.append(30.0)
        elif s == 9:
             # Hexagonal lattice but rotated 15 deg
            current_params = []
            for i, (q, r) in enumerate(init_grid_coords):
                cx, cy = grid_to_cart(q, r)
                rad = math.radians(15)
                nx = cx * math.cos(rad) - cy * math.sin(rad)
                ny = cx * math.sin(rad) + cy * math.cos(rad)
                current_params.append(nx)
                current_params.append(ny)
                current_params.append(15.0)
            current_params.append(15.0)
        else:
            # More random perturbations of the best found so far
            if s < 12:
                current_params = best_centers and [x for c in best_centers for x in c] + best_angles + [best_outer_angle]
                for i in range(N):
                    current_params[i * 3] += rng.uniform(-0.2, 0.2)
                    current_params[i * 3 + 1] += rng.uniform(-0.2, 0.2)
                    current_params[i * 3 + 2] += rng.uniform(-5.0, 5.0)
                current_params[36] += rng.uniform(-5.0, 5.0)
            else:
                current_params = init_params[:]
                for i in range(N):
                    current_params[i * 3] += rng.uniform(-2.0, 2.0)
                    current_params[i * 3 + 1] += rng.uniform(-2.0, 2.0)
                    current_params[i * 3 + 2] += rng.uniform(-30.0, 30.0)
                current_params[36] = rng.uniform(-30.0, 30.0)

        # Run Nelder-Mead
        try:
            res = minimize(
                objective,
                current_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
            )
        except Exception:
            continue

        if res.fun < objective(current_params):
            current_params = res.x
            # Verify validity
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
                    # Optional: Polish with SLSQP if time allows
                    if time.time() < deadline - 30.0:
                        try:
                            res2 = minimize(
                                objective,
                                current_params,
                                method='SLSQP',
                                options={'maxiter': 100, 'ftol': 1e-8}
                            )
                            if res2.fun < objective(current_params):
                                current_params = res2.x
                                centers_opt2 = []
                                angles_opt2 = []
                                for i in range(N):
                                    centers_opt2.append((current_params[i * 3], current_params[i * 3 + 1]))
                                    angles_opt2.append(current_params[i * 3 + 2])
                                outer_angle2 = current_params[36]
                                if check_overlap(centers_opt2, angles_opt2):
                                    L2 = enclosing_side_length(centers_opt2, angles_opt2, outer_angle2)
                                    if L2 < best_L:
                                        best_L = L2
                                        best_centers = centers_opt2
                                        best_angles = angles_opt2
                                        best_outer_angle = outer_angle2
                        except Exception:
                            pass

    # Final check and return
    if not check_overlap(best_centers, best_angles):
        # Fallback to initial valid one
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
