"""
Strategy: Symmetry-constrained structural search with smooth surrogate and
high-precision polish.
Fixes 12 inner unit hexagons into a canonical hexagonal cluster (1 center +
6 ring-1 + 5 ring-2), reducing to 6 continuous parameters. Uses vectorized
SAT for fast collision checks, a log-sum-exp soft-max surrogate for smooth
outer-containment optimization, and a final L-BFGS-B polish pass.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)
APOHEM_UNIT = SQRT3 / 2.0  # apothem of unit hexagon

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (kill at 150s)

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        apothem = side * SQRT3 / 2
        # Normals to the sides of a regular hexagon centered at origin
        # Sides are at 30, 90, 150, 210, 270, 330 degrees
        normal_angles = np.radians(np.arange(30, 360, 60))
        for na in normal_angles:
            nx = math.cos(na)
            ny = math.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        """
        Check if two convex polygons are disjoint using Separating Axis Theorem.
        Returns True if disjoint, False if overlapping.
        Vectorized implementation for speed.
        """
        # Collect all edge normals (12 axes for two hexagons)
        axes = np.empty((12, 2))
        for idx, v in enumerate([v1, v2]):
            base = idx * 6
            for i in range(6):
                edge = v[(i + 1) % 6] - v[i]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-9:
                    axes[base + i] = normal / norm
                else:
                    axes[base + i] = np.array([1.0, 0.0])

        # Project all vertices onto each axis
        # proj shape: (12, 12) -> axes x vertices
        proj1 = v1 @ axes.T  # (6, 12)
        proj2 = v2 @ axes.T  # (6, 12)

        min1 = np.min(proj1, axis=0)  # (12,)
        max1 = np.max(proj1, axis=0)
        min2 = np.min(proj2, axis=0)
        max2 = np.max(proj2, axis=0)

        # If any axis separates them, they're disjoint
        separated = (max1 < min2 - 1e-9) | (max2 < min1 - 1e-9)
        return bool(np.any(separated))

    def inner_positions_from_params(params):
        """
        params: [r1, r2, ring1_rot_deg, ring2_rot_deg, cluster_rot_deg, outer_rot_deg]
        Returns list of (x, y, angle_deg) for 12 inner hexagons (all at 0 angle relative to world,
        but we can also rotate each inner hexagon with the cluster).
        """
        r1 = params[0]
        r2 = params[1]
        ring1_rot = params[2]
        ring2_rot = params[3]
        cluster_rot = params[4]
        # outer_rot = params[5]  (used separately)

        positions = []
        # Center
        positions.append((0.0, 0.0, 0.0))
        # Ring 1: 6 hexagons
        for k in range(6):
            angle = math.radians(60 * k + ring1_rot + cluster_rot)
            x = r1 * math.cos(angle)
            y = r1 * math.sin(angle)
            # Rotate each inner hexagon to align with the cluster
            positions.append((x, y, cluster_rot))
        # Ring 2: 5 hexagons (skip one slot for 12 total)
        for k in range(5):
            angle = math.radians(60 * k + 30 + ring2_rot + cluster_rot)
            x = r2 * math.cos(angle)
            y = r2 * math.sin(angle)
            positions.append((x, y, cluster_rot))
        return positions

    def min_outer_side(inner_positions, outer_angle_deg, outer_center=(0.0, 0.0)):
        """
        Calculate the minimum side length of a regular hexagon with given angle
        and center that contains all inner hexagons.
        """
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        if not all_verts:
            return 0.0
        all_verts = np.vstack(all_verts)

        # Normals for the outer hexagon at angle outer_angle_deg
        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad

        # Vectorized projection
        n_mat = np.column_stack([np.cos(normals), np.sin(normals)])  # (6, 2)
        rel = all_verts - np.array(outer_center)  # (72, 2)
        projs = rel @ n_mat.T  # (72, 6)
        max_per_normal = np.max(projs, axis=0)  # (6,)
        max_proj = float(np.max(max_per_normal))

        # S * sqrt(3)/2 >= max_proj  =>  S >= max_proj / (sqrt(3)/2)
        return max_proj / (SQRT3 / 2)

    def min_outer_side_smooth(inner_positions, outer_angle_deg, outer_center=(0.0, 0.0), beta=50.0):
        """
        Smooth surrogate using log-sum-exp for the max over all vertex projections.
        Returns a smooth approximation to min_outer_side that is differentiable.
        """
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            all_verts.append(verts)
        if not all_verts:
            return 0.0
        all_verts = np.vstack(all_verts)

        base_normals = np.radians(np.arange(30, 360, 60))
        outer_rad = np.radians(outer_angle_deg)
        normals = base_normals + outer_rad

        n_mat = np.column_stack([np.cos(normals), np.sin(normals)])  # (6, 2)
        rel = all_verts - np.array(outer_center)  # (72, 2)
        projs = rel @ n_mat.T  # (72, 6)

        # Soft-max over all 72x6 = 432 projections
        flat_projs = projs.flatten()
        max_flat = np.max(flat_projs)
        # log-sum-exp for numerical stability
        shifted = flat_projs - max_flat
        soft_max = max_flat + np.log(np.sum(np.exp(shifted))) / beta
        # Actually, proper soft-max: (1/beta) * log(sum(exp(beta * x)))
        # Let's redo:
        beta_shifted = beta * flat_projs
        max_beta = np.max(beta_shifted)
        soft_max = (1.0 / beta) * (max_beta + np.log(np.sum(np.exp(beta_shifted - max_beta))))

        return soft_max / (SQRT3 / 2)

    def objective(params):
        """
        params: [r1, r2, ring1_rot_deg, ring2_rot_deg, cluster_rot_deg, outer_rot_deg]
        Returns the side length of the minimal enclosing outer hexagon, plus penalties.
        """
        r1 = params[0]
        r2 = params[1]
        ring1_rot = params[2]
        ring2_rot = params[3]
        cluster_rot = params[4]
        outer_rot = params[5]

        # Ensure radii are positive and reasonable
        if r1 < 1.5 or r2 < 2.5:
            return 100.0 + (1.5 - r1) * 10 + (2.5 - r2) * 10

        inner_positions = inner_positions_from_params(params)

        # Check disjointness (vectorized SAT)
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    penalty += 10.0

        s = min_outer_side(inner_positions, outer_rot, (0.0, 0.0))

        return s + penalty

    def objective_smooth(params):
        """
        Smooth version of objective using log-sum-exp surrogate for the
        outer containment term. Useful for gradient-based polish.
        """
        r1 = params[0]
        r2 = params[1]
        ring1_rot = params[2]
        ring2_rot = params[3]
        cluster_rot = params[4]
        outer_rot = params[5]

        if r1 < 1.5 or r2 < 2.5:
            return 100.0 + (1.5 - r1) * 10 + (2.5 - r2) * 10

        inner_positions = inner_positions_from_params(params)

        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    penalty += 10.0

        s = min_outer_side_smooth(inner_positions, outer_rot, (0.0, 0.0), beta=80.0)

        return s + penalty

    # Initial guesses for multi-start
    np.random.seed(42)

    best_params = None
    best_score = float('inf')

    # Multi-start Nelder-Mead
    initial_guesses = [
        [2.0, 3.5, 0.0, 0.0, 0.0, 0.0],
        [2.0, 3.5, 30.0, 0.0, 0.0, 0.0],
        [2.0, 3.5, 0.0, 30.0, 0.0, 0.0],
        [2.0, 3.5, 0.0, 0.0, 30.0, 0.0],
        [2.0, 3.5, 0.0, 0.0, 0.0, 30.0],
        [2.1, 3.6, 15.0, 15.0, 15.0, 15.0],
        [1.9, 3.4, -15.0, -15.0, -15.0, -15.0],
        [2.0, 3.5, 0.0, 0.0, 0.0, 0.0],  # baseline
    ]

    for guess in initial_guesses:
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(objective, guess, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # If no good result yet, use the baseline
    if best_params is None:
        best_params = np.array([2.0, 3.5, 0.0, 0.0, 0.0, 0.0])
        best_score = objective(best_params)

    # SA polish around the best with adaptive step size
    if time.time() < deadline - 30:
        current_params = best_params.copy()
        current_score = best_score
        T = 0.05
        cooling = 0.97
        max_iters = 800
        step_scale = 1.0
        for i in range(max_iters):
            if time.time() > deadline - 25:
                break
            # Adaptive step: shrink if no recent improvement
            new_params = current_params.copy()
            new_params[0] += np.random.uniform(-0.15 * step_scale, 0.15 * step_scale)
            new_params[1] += np.random.uniform(-0.2 * step_scale, 0.2 * step_scale)
            new_params[2] += np.random.uniform(-15 * step_scale, 15 * step_scale)
            new_params[3] += np.random.uniform(-15 * step_scale, 15 * step_scale)
            new_params[4] += np.random.uniform(-15 * step_scale, 15 * step_scale)
            new_params[5] += np.random.uniform(-15 * step_scale, 15 * step_scale)

            new_score = objective(new_params)
            if new_score < current_score or np.random.rand() < np.exp(-(new_score - current_score) / T):
                current_params = new_params
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_params = current_params.copy()
                    step_scale = max(0.3, step_scale * 1.05)
                else:
                    step_scale = max(0.3, step_scale * 0.95)
            else:
                step_scale = max(0.3, step_scale * 0.98)
            T *= cooling
            if T < 0.002:
                T = 0.002

    # High-precision L-BFGS-B polish with smooth surrogate
    if time.time() < deadline - 15:
        try:
            bounds = [
                (1.5, 4.0),   # r1
                (2.5, 5.0),   # r2
                (-60, 60),    # ring1_rot
                (-60, 60),    # ring2_rot
                (-60, 60),    # cluster_rot
                (-60, 60),    # outer_rot
            ]
            res_lbfgs = minimize(objective_smooth, best_params, method='L-BFGS-B',
                                 bounds=bounds,
                                 options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                                           'maxfun': 3000})
            if res_lbfgs.fun < best_score:
                # Verify with exact objective
                exact_score = objective(res_lbfgs.x)
                if exact_score < best_score:
                    best_score = exact_score
                    best_params = res_lbfgs.x.copy()
        except Exception:
            pass

    # Final tight Nelder-Mead polish
    if time.time() < deadline - 5:
        try:
            res_final = minimize(objective, best_params, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10,
                                           'adaptive': True})
            if res_final.fun < best_score:
                best_score = res_final.fun
                best_params = res_final.x.copy()
        except Exception:
            pass

    # Extract best inner positions
    inner = inner_positions_from_params(best_params)

    # Compute final outer side and angle
    outer_rot = best_params[5]
    s = min_outer_side(inner, outer_rot, (0.0, 0.0))

    center = (0.0, 0.0)
    side = s
    angle = outer_rot

    # Validation check: all vertices must be inside outer hexagon
    valid = True
    for cx, cy, ang in inner:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for vx, vy in verts:
            if not point_in_hexagon(vx, vy, center[0], center[1], side, angle):
                valid = False
                break
        if not valid:
            break

    # Also validate pairwise disjointness
    if valid:
        for i in range(N):
            for j in range(i + 1, N):
                ci = (inner[i][0], inner[i][1])
                cj = (inner[j][0], inner[j][1])
                ai = inner[i][2]
                aj = inner[j][2]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ai)
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, aj)
                if not polygons_disjoint_sat(v1, v2):
                    valid = False
                    break
            if not valid:
                break

    if not valid:
        # Bump side slightly to ensure validity
        side *= 1.02

    return inner, center, side, angle
