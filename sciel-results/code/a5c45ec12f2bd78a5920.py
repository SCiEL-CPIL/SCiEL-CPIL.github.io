"""Rewritten program: honeycomb lattice + differential evolution for 12 hex packing."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    # Precompute unit hexagon vertices (side=1, centered at origin, flat-top orientation)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])

    def hex_support(directions):
        """Compute support function of unit hexagon in given directions.
        directions: array of shape (6, 2) with 6 normal directions.
        Returns array of shape (6,) with max projection in each direction.
        """
        # proj[i, j] = vertex j dot direction i
        proj = unit_verts.T @ directions  # shape (6, 6)
        return proj.max(axis=1)  # shape (6,)

    def calc_side_length(positions, theta_deg, cx=0.0, cy=0.0):
        """Compute minimum side length of outer hexagon containing all inner hexagons.
        positions: list of (x, y) centers
        theta_deg: rotation of outer hexagon in degrees
        cx, cy: center of outer hexagon
        Returns: side length S
        """
        theta_rad = math.radians(theta_deg)
        # 6 outward normal directions for a regular hexagon (pointing outward from center)
        # For a hexagon with flat side facing up, normals are at 0, 60, 120, 180, 240, 300
        # But we need the normals to the sides. For a regular hexagon, the outward normals
        # to the 6 sides are at angles 0, 60, 120, 180, 240, 300 (if one vertex is at angle 0)
        # Actually, for a regular hexagon with a vertex at angle 0, the side normals are at 30, 90, 150, 210, 270, 330
        # Let's use: normals at 30 + k*60 for k=0..5, then add theta
        normals = np.array([
            [math.cos(math.radians(30 + k*60)), math.sin(math.radians(30 + k*60))]
            for k in range(6)
        ])
        # Rotate normals by theta
        cos_t = math.cos(theta_rad)
        sin_t = math.sin(theta_rad)
        rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        normals = normals @ rot.T  # shape (6, 2)

        # Support of unit hexagon in each normal direction
        h_unit = hex_support(normals)  # shape (6,)

        # For each normal direction, compute max over all inner hexagons of:
        # (center_i - outer_center) dot normal + h_unit
        max_proj = 0.0
        for k in range(6):
            nx, ny = normals[k]
            h = h_unit[k]
            max_in_dir = 0.0
            for (x, y) in positions:
                proj = (x - cx) * nx + (y - cy) * ny + h
                if proj > max_in_dir:
                    max_in_dir = proj
            if max_in_dir > max_proj:
                max_proj = max_in_dir

        # Side length S: for a regular hexagon, the distance from center to a side (apothem) = S * sqrt(3)/2
        # So S = max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / sqrt3

    def overlap_penalty(positions, eps=1e-9):
        """Compute penalty for overlapping hexagons.
        Two unit hexagons overlap if their centers are closer than 2.0 (in the worst case).
        Actually, for two identical regular hexagons of side 1, the minimum distance between
        centers for no overlap depends on relative orientation. For aligned hexagons, it's 2.0.
        For arbitrary orientations, it can be less. We use a conservative bound: if distance < 2.0,
        there's potential overlap.
        """
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = positions[i][0] - positions[j][0]
                dy = positions[i][1] - positions[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 2.0 - eps:
                    penalty += (2.0 - dist) ** 2
        return penalty

    def objective(params):
        """Objective function for optimization.
        params: [x0, y0, x1, y1, ..., x11, y11, theta]
        Returns: side length + penalty for overlaps
        """
        if time.time() > deadline:
            return 1e10

        positions = [(params[i], params[i+1]) for i in range(0, 2*N, 2)]
        theta = params[2*N]

        # Optimize outer center to minimize side length (use centroid as initial guess)
        cx0 = sum(x for x, y in positions) / N
        cy0 = sum(y for x, y in positions) / N

        S = calc_side_length(positions, theta, cx0, cy0)
        pen = overlap_penalty(positions)

        return S + 100.0 * pen

    # Initial arrangement: 3x4 honeycomb patch
    # Hexagon centers in a honeycomb lattice:
    # Row 0: x = 0, 2, 4, y = 0
    # Row 1: x = 1, 3, y = sqrt(3)
    # Row 2: x = 0, 2, 4, y = 2*sqrt(3)
    # Row 3: x = 1, 3, y = 3*sqrt(3)
    # That's 4+2+4+2 = 12 hexagons

    init_positions = []
    for row in range(4):
        y = row * sqrt3
        if row % 2 == 0:
            for col in range(4):
                x = col * 2.0
                init_positions.append((x, y))
        else:
            for col in range(2):
                x = col * 2.0 + 1.0
                init_positions.append((x, y))

    # Center the arrangement
    xs = [p[0] for p in init_positions]
    ys = [p[1] for p in init_positions]
    cx_off = (min(xs) + max(xs)) / 2.0
    cy_off = (min(ys) + max(ys)) / 2.0
    init_positions = [(x - cx_off, y - cy_off) for x, y in init_positions]

    # Build initial parameter vector
    init_params = []
    for x, y in init_positions:
        init_params.extend([x, y])
    init_params.append(0.0)  # theta

    best_params = init_params[:]
    best_score = objective(init_params)

    # Define bounds for differential evolution
    # Positions: roughly within [-5, 5] in each coordinate
    # Theta: within [-30, 30] degrees
    bounds = [(-5, 5)] * (2*N) + [(-30, 30)]

    # Try differential evolution for global optimization
    try:
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=200,
            tol=1e-6,
            polish=False,  # We'll polish separately
            updating='immediate',
            popsize=15
        )
        if result.fun < best_score:
            best_score = result.fun
            best_params = result.x.copy()
    except Exception as e:
        pass

    # Local refinement with Nelder-Mead
    if time.time() < deadline - 20:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Extract final positions and theta
    pos = [(best_params[i], best_params[i+1]) for i in range(0, 2*N, 2)]
    theta = best_params[2*N]

    # Optimize outer center more carefully
    def outer_side_only(cx, cy):
        return calc_side_length(pos, theta, cx, cy)

    cx0 = sum(x for x, y in pos) / N
    cy0 = sum(y for x, y in pos) / N

    try:
        center_res = minimize(
            lambda c: outer_side_only(c[0], c[1]),
            [cx0, cy0],
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
        )
        best_cx, best_cy = center_res.x
    except Exception:
        best_cx, best_cy = cx0, cy0

    final_side = calc_side_length(pos, theta, best_cx, best_cy)

    # Also try a few different outer rotations to see if we can do better
    # The theta in best_params is the outer hexagon rotation
    # Let's also try optimizing theta jointly with center
    def full_outer(params):
        theta_opt = params[0]
        cx = params[1]
        cy = params[2]
        return calc_side_length(pos, theta_opt, cx, cy)

    if time.time() < deadline - 10:
        try:
            outer_res = minimize(
                full_outer,
                [theta, best_cx, best_cy],
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if outer_res.fun < final_side:
                final_side = outer_res.fun
                theta = outer_res.x[0]
                best_cx, best_cy = outer_res.x[1], outer_res.x[2]
        except Exception:
            pass

    inner = []
    for x, y in pos:
        inner.append([x, y, 0.0])

    return inner, (best_cx, best_cy), final_side, theta
# EVOLVE-BLOCK-END
