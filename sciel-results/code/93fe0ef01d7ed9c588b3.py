"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0  # 150s limit, keep 10s margin

    def hex_vertices(cx, cy, side, angle_deg):
        """Get vertices of a regular hexagon with given center, side length, and rotation."""
        angle = math.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def hex_radius(side):
        """Circumradius (vertex distance) of a regular hexagon with given side length."""
        return side

    def dist_point_to_segment(px, py, x1, y1, x2, y2):
        """Distance from point (px,py) to segment (x1,y1)-(x2,y2)."""
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(px - x1, py - y1)
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        t = max(0.0, min(1.0, t))
        qx = x1 + t * dx
        qy = y1 + t * dy
        return math.hypot(px - qx, py - qy)

    def min_dist_hexagons(c1, a1, s1, c2, a2, s2):
        """
        Approximate minimum distance between two regular hexagons.
        Uses vertex-to-vertex and vertex-to-edge checks.
        c1, c2: (x,y) centers; a1, a2: angles in degrees; s1, s2: side lengths.
        Returns a lower bound on the distance (0 if overlapping).
        """
        v1 = hex_vertices(c1[0], c1[1], s1, a1)
        v2 = hex_vertices(c2[0], c2[1], s2, a2)

        min_d = float('inf')

        # Check all vertex-vertex pairs
        for p in v1:
            for q in v2:
                d = math.hypot(p[0] - q[0], p[1] - q[1])
                if d < min_d:
                    min_d = d

        # Check vertex of v1 to edges of v2
        for p in v1:
            for i in range(6):
                d = dist_point_to_segment(p[0], p[1], v2[i][0], v2[i][1], v2[(i+1)%6][0], v2[(i+1)%6][1])
                if d < min_d:
                    min_d = d

        # Check vertex of v2 to edges of v1
        for p in v2:
            for i in range(6):
                d = dist_point_to_segment(p[0], p[1], v1[i][0], v1[i][1], v1[(i+1)%6][0], v1[(i+1)%6][1])
                if d < min_d:
                    min_d = d

        return min_d

    def compute_outer_hex(params):
        """
        Given inner hexagon parameters (list of [x, y, angle_deg]),
        compute the minimal enclosing regular hexagon.
        Returns (center_x, center_y, side, angle_deg).
        We try all 6 orientations for the outer hexagon.
        """
        inner = params
        best_side = float('inf')
        best_center = (0.0, 0.0)
        best_angle = 0.0

        # Try different orientations for outer hexagon
        for outer_angle in [0, 15, 30, 45, 60, 90, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]:
            # For a given outer orientation, find minimal side length and center
            # such that all inner hexagons fit inside.
            # The outer hexagon with center (cx, cy), side s, angle a contains point (px, py)
            # if for all 6 half-planes defined by edges, the point is on the inside.

            # We'll use a simple approach: compute the max extent needed.
            # For a regular hexagon with side s and angle a, the distance from center to each
            # edge (inradius) is s * sqrt(3)/2.
            # A point (px, py) is inside if its projection onto each of the 6 inward normals
            # is <= inradius.

            inradius = math.sqrt(3) / 2  # for unit side, will scale by s

            # The 6 inward normals for a hexagon with angle a
            a_rad = math.radians(outer_angle)
            normals = []
            for i in range(6):
                # The outward normal for edge i is at angle a + i*pi/3 + pi/6
                # Actually, for a regular hexagon, the edges are perpendicular to directions
                # a + i*pi/3 for i=0..5 (these are the directions from center to midpoints of edges)
                # The inward normal for edge i points from edge toward center.
                # Let's define: edge i has outward normal direction theta_i = a + i*pi/3
                # A point is inside if dot(point - center, outward_normal_i) <= inradius * s
                theta_i = a_rad + i * math.pi / 3
                normals.append((math.cos(theta_i), math.sin(theta_i)))

            # For a given center (cx, cy) and side s, point (px,py) is inside if:
            # For all i: (px-cx)*nx_i + (py-cy)*ny_i <= s * sqrt(3)/2
            # Rearranging: s >= [(px-cx)*nx_i + (py-cy)*ny_i] / (sqrt(3)/2) for all i
            # And also s >= 0

            # We want to minimize s over (cx, cy). This is a convex problem in (cx, cy, s).
            # s >= max_i [(px_j - cx)*nx_i + (py_j - cy)*ny_i] / (sqrt(3)/2) for all j, i
            # = max_i [ (px_j*nx_i + py_j*ny_i) - (cx*nx_i + cy*ny_i) ] / (sqrt(3)/2)

            # For a fixed center, s = max over all (j, i) of [(px_j*nx_i + py_j*ny_i) - (cx*nx_i + cy*ny_i)] / (sqrt(3)/2)

            # To minimize s, we can try to find the center that equalizes the maximum constraints.
            # This is a linear programming problem, but let's use a simpler heuristic:
            # Try center at the centroid of all inner hexagon centers, and a few nearby points.

            # Get all vertices of all inner hexagons
            all_points = []
            for (x, y, ang) in inner:
                verts = hex_vertices(x, y, 1.0, ang)
                for v in verts:
                    all_points.append(v)
            all_points = np.array(all_points)

            if len(all_points) == 0:
                continue

            # Compute candidate centers
            centroid = all_points.mean(axis=0)

            # Try a grid of centers around the centroid
            best_s_for_angle = float('inf')
            best_cx, best_cy = centroid[0], centroid[1]

            # Use scipy to minimize: f(cx, cy) = max over all (point, normal) of (point·n - center·n) / (sqrt(3)/2)
            # This is a convex function, so we can use Nelder-Mead or similar.

            def objective(center_xy):
                cx, cy = center_xy
                max_val = 0.0
                for i in range(6):
                    nx, ny = normals[i]
                    # For each point, compute (px*nx + py*ny - cx*nx - cy*ny)
                    vals = all_points[:, 0] * nx + all_points[:, 1] * ny - cx * nx - cy * ny
                    m = vals.max()
                    if m > max_val:
                        max_val = m
                return max_val / (math.sqrt(3) / 2)

            # Minimize using Nelder-Mead
            x0 = np.array([centroid[0], centroid[1]])
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})

            s_candidate = res.fun
            if s_candidate < best_s_for_angle:
                best_s_for_angle = s_candidate
                best_cx, best_cy = res.x[0], res.x[1]

            if best_s_for_angle < best_side:
                best_side = best_s_for_angle
                best_center = (best_cx, best_cy)
                best_angle = outer_angle

        return best_center, best_side, best_angle

    # Initial layout: hexagonal grid
    # Place hexagons in a honeycomb pattern
    # For 12 hexagons, use a 3-4-3 arrangement or similar
    # Axial coordinates for a hex grid
    def axial_to_pixel(q, r):
        """Convert axial coordinates to pixel coordinates for flat-top hexagons."""
        x = 2.0 * q + r
        y = math.sqrt(3) * r
        return (x, y)

    # 12 hexagons in a hexagonal arrangement:
    # Center + 6 neighbors + 5 more
    axial_coords = [(0, 0),
                    (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
                    (2, 0), (2, -1), (2, -2), (1, -2), (0, -2)]

    inner = []
    for (q, r) in axial_coords:
        x, y = axial_to_pixel(q, r)
        inner.append([x, y, 0.0])

    # Optimize positions and angles
    def flatten_params(params):
        """Flatten list of [x, y, angle] to a flat array."""
        return np.array([p[0] for p in params] + [p[1] for p in params] + [p[2] for p in params])

    def unflatten_params(arr):
        """Unflatten array to list of [x, y, angle]."""
        n = len(arr) // 3
        params = []
        for i in range(n):
            params.append([arr[3*i], arr[3*i+1], arr[3*i+2]])
        return params

    def objective_flat(arr):
        """Compute the outer hexagon side length for given inner parameters."""
        if time.time() > deadline:
            return float('inf')
        params = unflatten_params(arr)
        _, side, _ = compute_outer_hex(params)
        return side

    x0 = flatten_params(inner)

    # Use Nelder-Mead for local optimization
    best_x = x0.copy()
    best_val = objective_flat(x0)

    # Multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline:
            break

        if restart == 0:
            x_init = x0.copy()
        else:
            # Perturb the best solution
            x_init = best_x + np.random.randn(len(x0)) * 0.5

        res = minimize(objective_flat, x_init, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})

        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x

    # Final result
    inner_final = unflatten_params(best_x)
    center, side, angle = compute_outer_hex(inner_final)

    # Ensure side is positive
    if side <= 0:
        side = 40.0

    return inner_final, center, side, angle
# EVOLVE-BLOCK-END

