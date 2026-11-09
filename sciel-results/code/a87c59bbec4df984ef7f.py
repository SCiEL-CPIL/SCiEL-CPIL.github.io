"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    sqrt3 = math.sqrt(3.0)

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

    def hexagon_separation(vi, vj, eps=1e-9):
        """
        Check if two convex hexagons (given as lists of 6 CCW vertices)
        are separated (non-overlapping) using the Separating Axis Theorem.
        Returns True if separated (no overlap), False if overlapping.
        """
        # For two convex polygons, check all edge normals of both polygons
        for poly in (vi, vj):
            for k in range(6):
                x1, y1 = poly[k]
                x2, y2 = poly[(k + 1) % 6]
                # Edge vector
                ex, ey = x2 - x1, y2 - y1
                # Normal (perpendicular, pointing outward for CCW polygon)
                nx, ny = ey, -ex
                # Project both polygons onto this normal
                min_i, max_i = float('inf'), float('-inf')
                min_j, max_j = float('inf'), float('-inf')
                for px, py in vi:
                    proj = px * nx + py * ny
                    if proj < min_i:
                        min_i = proj
                    if proj > max_i:
                        max_i = proj
                for px, py in vj:
                    proj = px * nx + py * ny
                    if proj < min_j:
                        min_j = proj
                    if proj > max_j:
                        max_j = proj
                # Check for gap
                if max_i < min_j - eps or max_j < min_i - eps:
                    # Found a separating axis -> polygons are separated
                    return True
        return False

    def check_overlap(centers, angles, eps=1e-6):
        """Check if any two inner hexagons overlap."""
        vertices_list = [get_hex_vertices(centers[i][0], centers[i][1], angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                # Quick distance check
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist_sq = dx * dx + dy * dy
                # If centers are very far apart, skip detailed check
                if dist_sq > 16.0:
                    continue
                if not hexagon_separation(vertices_list[i], vertices_list[j], eps):
                    return False
        return True

    def enclosing_side_length(centers, angles, outer_angle=0.0):
        """
        Compute the minimal side length of a regular hexagon (centered at origin,
        with given rotation) that contains all inner unit hexagons.
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

        # max_extent is the apothem of the outer hexagon
        return 2.0 * max_extent / sqrt3

    # Initialize with a valid hexagonal lattice configuration
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

    # Verify this initial config is valid
    if not check_overlap(centers, angles):
        centers = [(4.0 * k, 0.0) for k in range(-(N // 2), N - N // 2)]
        angles = [0.0] * N

    best_L = enclosing_side_length(centers, angles, 0.0)
    best_centers = [c[:] for c in centers]
    best_angles = angles[:]
    best_outer_angle = 0.0

    def unpack(params):
        """Unpack 37 parameters into centers, angles, outer_angle."""
        c = [(params[i * 3], params[i * 3 + 1]) for i in range(N)]
        a = [params[i * 3 + 2] for i in range(N)]
        oa = params[36]
        return c, a, oa

    def pack(centers, angles, outer_angle):
        """Pack centers, angles, outer_angle into 37 parameters."""
        p = []
        for i in range(N):
            p.extend([centers[i][0], centers[i][1], angles[i]])
        p.append(outer_angle)
        return p

    # Phase 1: Nelder-Mead for coarse global optimization
    def objective_nm(params):
        if time.time() > deadline - 15.0:
            return 1e6
        c, a, oa = unpack(params)
        if not check_overlap(c, a):
            return 1e6
        return enclosing_side_length(c, a, oa)

    init_params = pack(centers, angles, best_outer_angle)

    if time.time() < deadline - 20.0:
        try:
            res = minimize(
                objective_nm,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True}
            )
            if res.fun < best_L and res.fun < 1e5:
                c, a, oa = unpack(res.x)
                if check_overlap(c, a):
                    L = enclosing_side_length(c, a, oa)
                    if L < best_L:
                        best_L = L
                        best_centers = c
                        best_angles = a
                        best_outer_angle = oa
        except Exception:
            pass

    # Phase 2: SLSQP for fine local optimization with smooth constraints
    if time.time() < deadline - 10.0:
        # SLSQP objective: minimize outer side length
        def objective_slsqp(params):
            if time.time() > deadline - 5.0:
                return 1e6
            c, a, oa = unpack(params)
            return enclosing_side_length(c, a, oa)

        # Constraints: for each pair (i,j), ensure minimum separation
        # Use the SAT-based separation distance as constraint
        def separation_distance(vi, vj):
            """Compute minimum separation distance between two convex hexagons."""
            min_sep = float('inf')
            for poly in (vi, vj):
                for k in range(6):
                    x1, y1 = poly[k]
                    x2, y2 = poly[(k + 1) % 6]
                    ex, ey = x2 - x1, y2 - y1
                    nx, ny = ey, -ex
                    # Normalize normal
                    norm = math.sqrt(nx * nx + ny * ny)
                    if norm < 1e-12:
                        continue
                    nx /= norm
                    ny /= norm
                    min_i, max_i = float('inf'), float('-inf')
                    min_j, max_j = float('inf'), float('-inf')
                    for px, py in vi:
                        proj = px * nx + py * ny
                        if proj < min_i:
                            min_i = proj
                        if proj > max_i:
                            max_i = proj
                    for px, py in vj:
                        proj = px * nx + py * ny
                        if proj < min_j:
                            min_j = proj
                        if proj > max_j:
                            max_j = proj
                    # Separation along this axis
                    sep = min(max_i - min_j, max_j - min_i)
                    if sep < min_sep:
                        min_sep = sep
            return min_sep

        def make_constraint(i, j):
            def constraint(params):
                if time.time() > deadline - 5.0:
                    return 1.0  # Don't violate
                c, a, oa = unpack(params)
                vi = get_hex_vertices(c[i][0], c[i][1], a[i])
                vj = get_hex_vertices(c[j][0], c[j][1], a[j])
                # Require separation >= 0 (touching is OK)
                return separation_distance(vi, vj)
            return constraint

        constraints = []
        for i in range(N):
            for j in range(i + 1, N):
                constraints.append({'type': 'ineq', 'fun': make_constraint(i, j)})

        bounds = [(-8, 8), (-8, 8), (-180, 180)] * N + [(-30, 30)]

        current_params = pack(best_centers, best_angles, best_outer_angle)

        try:
            res2 = minimize(
                objective_slsqp,
                current_params,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
            )
            if res2.fun < best_L and res2.fun < 1e5:
                c, a, oa = unpack(res2.x)
                if check_overlap(c, a):
                    L = enclosing_side_length(c, a, oa)
                    if L < best_L:
                        best_L = L
                        best_centers = c
                        best_angles = a
                        best_outer_angle = oa
        except Exception:
            pass

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
