import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580  # 50s margin from 630s limit

    # Base hexagon vertices for side 1, angle 0
    angles_base = np.deg2rad(np.arange(0, 360, 60))
    base_verts = np.column_stack([np.cos(angles_base), np.sin(angles_base)])

    # Outer hexagon normals (for containment check)
    outer_normal_angles = np.deg2rad(np.arange(30, 360, 60))
    outer_normals = np.column_stack([np.cos(outer_normal_angles), np.sin(outer_normal_angles)])

    def get_normals(verts):
        """Get edge normals for a hexagon (6,2) vertex array."""
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        return normals / norms

    def min_gap(v1, v2):
        """Compute minimum gap between two hexagons using SAT."""
        n1 = get_normals(v1)
        n2 = get_normals(v2)
        all_normals = np.vstack([n1, n2])

        proj1 = v1 @ all_normals.T
        proj2 = v2 @ all_normals.T

        min1 = np.min(proj1, axis=1)
        max1 = np.max(proj1, axis=1)
        min2 = np.min(proj2, axis=1)
        max2 = np.max(proj2, axis=1)

        gaps = np.maximum(min2 - max1, min1 - max2)
        return np.min(gaps)

    def compute_vertices(xy, angles):
        """Compute (11, 6, 2) vertices from centers and angles."""
        cos_a = np.cos(np.deg2rad(angles))
        sin_a = np.sin(np.deg2rad(angles))
        vx = base_verts[:, 0]
        vy = base_verts[:, 1]
        rot_vx = cos_a[:, None] * vx[None, :] - sin_a[:, None] * vy[None, :]
        rot_vy = sin_a[:, None] * vx[None, :] + cos_a[:, None] * vy[None, :]
        verts_x = xy[:, 0:1] + rot_vx
        verts_y = xy[:, 1:2] + rot_vy
        return np.stack([verts_x, verts_y], axis=-1)

    def evaluate(params):
        """Objective: minimize outer hexagon side length with penalty for overlap."""
        if time.time() > deadline:
            return 1e10

        xy = params[:, :2]
        angles = params[:, 2]
        verts = compute_vertices(xy, angles)
        flat_verts = verts.reshape(-1, 2)

        # Containment: project onto outer normals
        projections = flat_verts @ outer_normals.T
        max_proj = np.max(projections, axis=1)
        global_max_proj = np.max(max_proj)
        S = 2.0 * global_max_proj / np.sqrt(3)

        # Separation: check all pairs
        min_gap_val = np.inf
        for i in range(11):
            for j in range(i + 1, 11):
                gap = min_gap(verts[i], verts[j])
                if gap < min_gap_val:
                    min_gap_val = gap

        penalty = 0.0
        if min_gap_val < 1e-6:
            penalty = (1e-6 - min_gap_val) ** 2 * 10000.0

        return S + penalty

    def is_valid(inner_data, outer_side):
        """Full validation check."""
        if not np.all(np.isfinite(inner_data)):
            return False
        if outer_side < np.sqrt(11) - 1e-9:
            return False

        xy = inner_data[:, :2]
        angles = inner_data[:, 2]
        verts = compute_vertices(xy, angles)

        # Check edge lengths
        for i in range(11):
            for j in range(6):
                p1 = verts[i, j]
                p2 = verts[i, (j + 1) % 6]
                edge_len = np.linalg.norm(p2 - p1)
                if abs(edge_len - 1.0) > 1e-8:
                    return False

        # Check containment
        flat_verts = verts.reshape(-1, 2)
        projections = flat_verts @ outer_normals.T
        max_proj = np.max(projections, axis=1)
        pure_S = 2.0 * np.max(max_proj) / np.sqrt(3)
        if pure_S > outer_side + 1e-6:
            return False

        # Check separation
        min_gap_val = np.inf
        for i in range(11):
            for j in range(i + 1, 11):
                gap = min_gap(verts[i], verts[j])
                if gap < min_gap_val:
                    min_gap_val = gap
        if min_gap_val < 1e-6:
            return False

        return True

    # Initial guess: honeycomb-like arrangement
    np.random.seed(42)
    x0 = np.zeros(11)
    y0 = np.zeros(11)
    a0 = np.zeros(11)

    # Center
    x0[0] = 0.0
    y0[0] = 0.0

    # Ring 1: 6 hexagons at distance sqrt(3)
    d = np.sqrt(3)
    for k in range(6):
        ang = np.deg2rad(k * 60)
        x0[1 + k] = d * np.cos(ang)
        y0[1 + k] = d * np.sin(ang)

    # Ring 2: 4 hexagons in the gaps
    for k in range(4):
        ang = np.deg2rad(k * 90 + 45)
        x0[7 + k] = 2.2 * np.cos(ang)
        y0[7 + k] = 2.2 * np.sin(ang)

    params0 = np.column_stack([x0, y0, a0])

    bounds = [(-4, 4), (-4, 4), (0, 360)] * 11

    best_S = 1e10
    best_params = params0.copy()

    # Phase 1: Differential Evolution
    try:
        res = differential_evolution(
            evaluate, bounds, seed=42, maxiter=800, popsize=15,
            tol=1e-4, polish=False, init='sobol'
        )
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x.copy()
    except Exception:
        pass

    # Phase 2: Nelder-Mead refinement
    if time.time() < deadline - 60:
        try:
            res_nm = minimize(
                evaluate, best_params, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
            )
            if res_nm.fun < best_S:
                best_S = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception:
            pass

    # Phase 3: L-BFGS-B with smooth penalty for final polish
    if time.time() < deadline - 30:
        def smooth_objective(params):
            if time.time() > deadline:
                return 1e10
            xy = params[:, :2]
            angles = params[:, 2]
            verts = compute_vertices(xy, angles)
            flat_verts = verts.reshape(-1, 2)
            projections = flat_verts @ outer_normals.T
            max_proj = np.max(projections, axis=1)
            S = 2.0 * np.max(max_proj) / np.sqrt(3)

            # Smooth penalty using soft-min for gaps
            min_gap_val = np.inf
            for i in range(11):
                for j in range(i + 1, 11):
                    gap = min_gap(verts[i], verts[j])
                    if gap < min_gap_val:
                        min_gap_val = gap

            # Soft penalty: exp(-gap * 1000) is smooth
            if min_gap_val < 1e-4:
                penalty = np.exp(-min_gap_val * 5000) * 10.0
            else:
                penalty = 0.0

            return S + penalty

        try:
            res_lbfgs = minimize(
                smooth_objective, best_params, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res_lbfgs.fun < best_S:
                best_S = res_lbfgs.fun
                best_params = res_lbfgs.x.copy()
        except Exception:
            pass

    # Final validation
    xy = best_params[:, :2]
    angles = best_params[:, 2]
    verts = compute_vertices(xy, angles)
    flat_verts = verts.reshape(-1, 2)
    projections = flat_verts @ outer_normals.T
    max_proj = np.max(projections, axis=1)
    pure_S = 2.0 * np.max(max_proj) / np.sqrt(3)

    valid = is_valid(best_params, pure_S)

    if not valid:
        # Fallback to known valid packing
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length

    inner_hex_data = best_params
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(pure_S)

    return inner_hex_data, outer_hex_data, outer_hex_side_length
