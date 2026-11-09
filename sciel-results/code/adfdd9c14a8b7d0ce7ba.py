"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import random
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0

    sqrt3 = math.sqrt(3.0)
    rng = random.Random(12345)

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
        for p in poly1:
            if point_in_convex(p[0], p[1], poly2):
                return True
        for p in poly2:
            if point_in_convex(p[0], p[1], poly1):
                return True
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
        for i in range(N):
            vi = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            for j in range(i + 1, N):
                vj = get_hex_vertices(centers[j][0], centers[j][1], angles[j])
                if polygons_intersect(vi, vj):
                    return False
        return True

    def enclosing_side_length(centers, angles, outer_angle=0.0):
        """Compute minimal side length of outer hexagon containing all inner hexagons."""
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

        if not check_overlap(centers_opt, angles_opt):
            L = enclosing_side_length(centers_opt, angles_opt, outer_angle)
            violation = 0.0
            for i in range(N):
                vi = get_hex_vertices(centers_opt[i][0], centers_opt[i][1], angles_opt[i])
                for j in range(i + 1, N):
                    vj = get_hex_vertices(centers_opt[j][0], centers_opt[j][1], angles_opt[j])
                    for p in vi:
                        if point_in_convex(p[0], p[1], vj):
                            violation += 1.0
                    for p in vj:
                        if point_in_convex(p[0], p[1], vi):
                            violation += 1.0
            return 1000.0 + L + 10.0 * violation

        return enclosing_side_length(centers_opt, angles_opt, outer_angle)

    init_params = []
    for i in range(N):
        init_params.append(centers[i][0])
        init_params.append(centers[i][1])
        init_params.append(angles[i])
    init_params.append(0.0)

    # Multi-start with different strategies
    n_starts = 12
    for s in range(n_starts):
        if time.time() > deadline - 10.0:
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
            # Random configuration
            current_params = []
            for i in range(N):
                current_params.append(rng.uniform(-3, 3))
                current_params.append(rng.uniform(-3, 3))
                current_params.append(rng.uniform(-60, 60))
            current_params.append(rng.uniform(-60, 60))
        elif s == 6:
            # Compact arrangement - try to fit in a circle
            current_params = []
            for i in range(N):
                angle = 2 * math.pi * i / N
                r = 2.5
                current_params.append(r * math.cos(angle))
                current_params.append(r * math.sin(angle))
                current_params.append(rng.uniform(-10, 10))
            current_params.append(0.0)
        elif s == 7:
            # Two rows of 6
            current_params = []
            for i in range(6):
                x = -4.5 + i * 1.8
                y = -1.2
                current_params.append(x)
                current_params.append(y)
                current_params.append(0.0)
            for i in range(6):
                x = -4.5 + i * 1.8
                y = 1.2
                current_params.append(x)
                current_params.append(y)
                current_params.append(0.0)
            current_params.append(0.0)
        elif s == 8:
            # Hexagonal ring with center
            current_params = []
            # Center
            current_params.append(0.0)
            current_params.append(0.0)
            current_params.append(0.0)
            # Ring of 6
            for i in range(6):
                angle = 2 * math.pi * i / 6
                r = 3.0
                current_params.append(r * math.cos(angle))
                current_params.append(r * math.sin(angle))
                current_params.append(0.0)
            # Outer 5
            for i in range(5):
                angle = 2 * math.pi * i / 5 + 0.3
                r = 5.0
                current_params.append(r * math.cos(angle))
                current_params.append(r * math.sin(angle))
                current_params.append(0.0)
            current_params.append(0.0)
        elif s == 9:
            # Try with outer angle at 30 degrees
            current_params = init_params[:]
            current_params[36] = 30.0
        elif s == 10:
            # Perturbed hexagonal lattice with rotations
            current_params = init_params[:]
            for i in range(N):
                current_params[i * 3 + 2] = rng.uniform(-20, 20)
            current_params[36] = rng.uniform(-10, 10)
        else:
            # Random
            current_params = []
            for i in range(N):
                current_params.append(rng.uniform(-4, 4))
                current_params.append(rng.uniform(-4, 4))
                current_params.append(rng.uniform(-60, 60))
            current_params.append(rng.uniform(-60, 60))

        # Simple local search: iteratively improve
        improved = True
        max_iter = 200
        step_size = 0.5
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            if not improved:
                break
            improved = False
            best_obj = objective(current_params)
            if best_obj > 1000:
                break

            # Try small perturbations
            for _ in range(20):
                trial = current_params[:]
                idx = rng.randint(0, 35)
                trial[idx] += rng.uniform(-step_size, step_size)
                obj = objective(trial)
                if obj < best_obj - 1e-6:
                    current_params = trial
                    best_obj = obj
                    improved = True
                    step_size = max(0.1, step_size * 0.9)
                    break

            if not improved:
                step_size *= 0.8

        # Evaluate final configuration
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
