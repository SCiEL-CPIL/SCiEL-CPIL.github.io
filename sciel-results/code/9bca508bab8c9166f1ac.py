"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Pre-compute unit hexagon vertices (angle 0, centered at origin)
    unit_verts = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    # Edges of unit hexagon
    unit_edges = np.array([[math.sin(math.radians(a)), -math.cos(math.radians(a))] for a in range(0, 360, 60)])
    # Edge normals (outward)
    unit_normals = unit_edges.copy()
    unit_normals = unit_normals / np.linalg.norm(unit_normals, axis=1, keepdims=True)

    def get_rotated_verts(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return unit_verts @ R.T + np.array([cx, cy])

    def compute_outer_side(centers, angles, outer_cx, outer_cy, outer_angle_deg):
        """
        For a regular hexagon centered at (outer_cx, outer_cy) with rotation outer_angle_deg,
        find the minimum side length to contain all inner hexagons.
        A regular hexagon of side s has support function s * sqrt(3)/2 in the 6 normal directions.
        So we need s = max over all inner hex vertices and all 6 outer normals of the projection.
        """
        outer_angle_rad = math.radians(outer_angle_deg)
        outer_normals = []
        for k in range(6):
            a = outer_angle_rad + k * math.pi / 3
            outer_normals.append(np.array([math.cos(a), math.sin(a)]))
        outer_normals = np.array(outer_normals)  # (6, 2)

        max_proj = 0.0
        for i in range(N):
            verts = get_rotated_verts(centers[i], angles[i], 0.0)
            rel = verts - np.array([outer_cx, outer_cy])
            # Project onto all 6 outer normals
            projs = rel @ outer_normals.T  # (6, 6)
            max_proj = max(max_proj, np.max(projs))
        return 2.0 * max_proj / math.sqrt(3)

    def pair_gap(verts1, verts2):
        """
        Compute the maximum gap between two convex polygons using SAT.
        gap > 0 means separated, gap < 0 means overlapping.
        """
        max_gap = -float('inf')
        # Use edge normals from both polygons
        for v in [verts1, verts2]:
            for i in range(6):
                e = v[(i+1) % 6] - v[i]
                n = np.array([-e[1], e[0]])
                norm = np.linalg.norm(n)
                if norm < 1e-12:
                    continue
                n = n / norm
                proj1 = np.dot(verts1, n)
                proj2 = np.dot(verts2, n)
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    gap = 0.0
                if gap > max_gap:
                    max_gap = gap
        return max_gap

    def objective(params):
        # params: 24 (inner centers) + 12 (inner angles in rad) + 3 (outer cx, cy, angle rad) = 39
        centers = params[:24].reshape(N, 2)
        inner_angles = params[24:36]  # radians
        outer_cx, outer_cy, outer_angle = params[36], params[37], params[38]

        # Compute side length
        side = compute_outer_side(centers, inner_angles, outer_cx, outer_cy, outer_angle)

        # Compute non-overlap penalty using smooth approximation
        # Use the actual gap: if gap < 0, they overlap
        penalty = 0.0
        verts_list = [get_rotated_verts(centers[i, 0], centers[i, 1], math.degrees(inner_angles[i])) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                gap = pair_gap(verts_list[i], verts_list[j])
                if gap < 0:
                    penalty += gap * gap
        return side + 1000.0 * penalty

    # --- Initial guess: honeycomb lattice ---
    # For 12 hexagons, a good arrangement is a "flower" pattern or a compact cluster.
    # Let's try a 3x4 hexagonal grid centered at origin.
    sqrt3 = math.sqrt(3)
    # Hexagonal grid: row r, column c
    # For row r (0-indexed), x offset = (c + (r % 2) * 0.5) * sqrt3
    # y = r * 1.5
    centers_init = np.zeros((N, 2))
    angles_init = np.zeros(N)
    # 3 columns, 4 rows
    idx = 0
    for r in range(4):
        for c in range(3):
            x = (c + (r % 2) * 0.5) * sqrt3
            y = r * 1.5
            centers_init[idx, 0] = x
            centers_init[idx, 1] = y
            idx += 1
    # Center the grid
    centers_init -= centers_init.mean(axis=0)

    x0 = np.concatenate([centers_init.flatten(), angles_init, [0.0, 0.0, 0.0]])

    # Run optimization
    best_side = float('inf')
    best_params = x0.copy()
    best_valid = False

    try:
        res = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-6, 'maxfun': 5000}
        )

        if time.time() < deadline:
            # Evaluate result
            centers = res.x[:24].reshape(N, 2)
            inner_angles_rad = res.x[24:36]
            outer_cx, outer_cy, outer_angle_rad = res.x[36], res.x[37], res.x[38]

            side = compute_outer_side(centers, inner_angles_rad, outer_cx, outer_cy, outer_angle_rad)

            # Check validity
            verts_list = [get_rotated_verts(centers[i, 0], centers[i, 1], math.degrees(inner_angles_rad[i])) for i in range(N)]
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    gap = pair_gap(verts_list[i], verts_list[j])
                    if gap < -1e-6:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                best_side = side
                best_params = res.x.copy()
                best_valid = True

    except Exception as e:
        pass

    # Fallback: use the initial honeycomb if optimization failed
    if not best_valid:
        centers = centers_init
        inner_angles_rad = angles_init
        outer_cx, outer_cy, outer_angle_rad = 0.0, 0.0, 0.0
        side = compute_outer_side(centers, inner_angles_rad, outer_cx, outer_cy, outer_angle_rad)
        best_side = side
        best_valid = True

    # Construct return value
    # inner: list of (x, y, angle_degrees)
    centers = best_params[:24].reshape(N, 2)
    inner_angles_rad = best_params[24:36]
    outer_cx, outer_cy, outer_angle_rad = best_params[36], best_params[37], best_params[38]

    inner = []
    for i in range(N):
        inner.append([float(centers[i, 0]), float(centers[i, 1]), float(math.degrees(inner_angles_rad[i]))])

    final_side = float(best_side * 1.001 + 1e-5)  # Tiny safety margin

    return inner, (float(outer_cx), float(outer_cy)), final_side, float(math.degrees(outer_angle_rad))
# EVOLVE-BLOCK-END
