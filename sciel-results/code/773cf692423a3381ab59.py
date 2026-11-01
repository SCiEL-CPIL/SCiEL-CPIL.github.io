import numpy as np
from scipy.optimize import minimize
import time
import itertools

N = 12
SQRT3 = np.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def hex_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        norms = np.where(norms < 1e-12, 1.0, norms)
        normals = normals / norms[:, np.newaxis]
        return normals

    def penetration_depth(v1, v2):
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])
        p1 = v1 @ normals.T
        p2 = v2 @ normals.T
        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)
        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        if np.min(overlaps) <= 1e-9:
            return 0.0
        return np.min(overlaps)

    def inside_violation(inner_verts, outer_verts):
        outer_normals = get_normals(outer_verts)
        p_inner = inner_verts @ outer_normals.T
        p_outer_max = (outer_verts @ outer_normals.T).max(axis=0)
        p_inner_max = p_inner.max(axis=0)
        viol = p_inner_max - p_outer_max
        return max(0.0, np.max(viol))

    def caging_objective(params, alpha_rad, theta_rad):
        """
        Caging formulation: minimize outer side length s such that
        all 12 inner hexagons (with shared rotation alpha) fit inside
        a regular hexagon of side s (rotation theta).
        
        Inner hexagons are treated as a rigid cluster. We optimize their
        positions (relative to outer center at origin) and s.
        
        params: [x1, y1, ..., x12, y12, s]
        """
        if time.time() > deadline:
            return 1e10

        positions = params[:2 * N].reshape(N, 2)
        s = params[2 * N]

        if s < 2.0:
            return 1e6

        outer_verts = hex_vertices(0, 0, s, theta_rad)

        total_penalty = 0.0
        inner_verts_list = []

        for i in range(N):
            x, y = positions[i]
            verts = hex_vertices(x, y, 1.0, alpha_rad)
            inner_verts_list.append(verts)
            viol = inside_violation(verts, outer_verts)
            total_penalty += viol * 100.0

        for i in range(N):
            for j in range(i + 1, N):
                pd = penetration_depth(inner_verts_list[i], inner_verts_list[j])
                total_penalty += pd * 100.0

        return s + total_penalty

    # Build initial configurations
    def make_flower_init():
        d1 = SQRT3
        d2 = 2 * SQRT3
        pos = []
        pos.append([0.0, 0.0])
        for k in range(6):
            a = np.radians(60 * k)
            pos.append([d1 * np.cos(a), d1 * np.sin(a)])
        for k in range(5):
            a = np.radians(60 * k)
            pos.append([d2 * np.cos(a), d2 * np.sin(a)])
        return np.array(pos)

    def make_ring_init():
        """Two concentric rings of 6."""
        d1 = SQRT3
        d2 = 2 * SQRT3
        pos = []
        for k in range(6):
            a = np.radians(60 * k)
            pos.append([d1 * np.cos(a), d1 * np.sin(a)])
        for k in range(6):
            a = np.radians(30 + 60 * k)
            pos.append([d2 * np.cos(a), d2 * np.sin(a)])
        return np.array(pos)

    def make_honeycomb_init():
        """3x4 honeycomb-like arrangement."""
        pos = []
        dx = 1.5
        dy = SQRT3
        for row in range(3):
            for col in range(4):
                x = col * dx + (row % 2) * dx / 2
                y = row * dy
                pos.append([x - 2.25, y - dy])
        return np.array(pos)

    inits = [make_flower_init(), make_ring_init(), make_honeycomb_init()]

    best_s = np.inf
    best_params = None
    best_alpha = 0.0
    best_theta = 0.0

    # Grid over rotations
    alphas = np.radians([0, 30])
    thetas = np.radians([0, 30])

    for init_idx, init_pos in enumerate(inits):
        if time.time() > deadline - 15:
            break
        for alpha in alphas:
            if time.time() > deadline - 10:
                break
            for theta in thetas:
                if time.time() > deadline - 5:
                    break

                init_s = 4.5
                x0 = np.concatenate([init_pos.flatten(), [init_s]])

                try:
                    res = minimize(
                        caging_objective,
                        x0,
                        args=(alpha, theta),
                        method='L-BFGS-B',
                        options={'maxiter': 300, 'ftol': 1e-8, 'gtol': 1e-6}
                    )

                    pos = res.x[:2 * N].reshape(N, 2)
                    s = res.x[2 * N]

                    outer_verts = hex_vertices(0, 0, s, theta)
                    valid = True
                    inner_verts_list = []
                    for i in range(N):
                        verts = hex_vertices(pos[i, 0], pos[i, 1], 1.0, alpha)
                        inner_verts_list.append(verts)
                        if inside_violation(verts, outer_verts) > 1e-5:
                            valid = False
                            break

                    if valid:
                        for i in range(N):
                            for j in range(i + 1, N):
                                if penetration_depth(inner_verts_list[i], inner_verts_list[j]) > 1e-5:
                                    valid = False
                                    break
                            if not valid:
                                break

                    if valid and s < best_s:
                        best_s = s
                        best_params = res.x.copy()
                        best_alpha = alpha
                        best_theta = theta

                except Exception:
                    pass

    if best_params is None:
        best_s = 5.0
        best_params = np.concatenate([make_flower_init().flatten(), [best_s]])
        best_alpha = 0.0
        best_theta = 0.0

    final_pos = best_params[:2 * N].reshape(N, 2)
    final_s = best_params[2 * N]

    inner = [[final_pos[i, 0], final_pos[i, 1], np.degrees(best_alpha)] for i in range(N)]
    outer_center = (0.0, 0.0)
    outer_side = final_s
    outer_angle_deg = np.degrees(best_theta)

    return inner, outer_center, outer_side, outer_angle_deg
