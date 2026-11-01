"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
from itertools import combinations

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill

    # Pre-compute constants for speed
    NORMALS = np.array([
        [np.cos(np.deg2rad(30 + k*60)), np.sin(np.deg2rad(30 + k*60))]
        for k in range(6)
    ])
    COS30 = np.sqrt(3) / 2
    VERTS_LOCAL = np.array([
        [1.0, 0.0], [0.5, np.sqrt(3)/2], [-0.5, np.sqrt(3)/2],
        [-1.0, 0.0], [-0.5, -np.sqrt(3)/2], [0.5, -np.sqrt(3)/2]
    ])

    def hexagon_vertices_fast(x, y, ang_deg):
        ang = np.deg2rad(ang_deg)
        c = np.cos(ang)
        s = np.sin(ang)
        R_mat = np.array([[c, -s], [s, c]])
        verts_rot = VERTS_LOCAL @ R_mat.T
        return verts_rot + np.array([x, y])

    def point_in_hexagon_fast(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        ang = np.deg2rad(-angle_deg)
        local_x = dx * np.cos(ang) - dy * np.sin(ang)
        local_y = dx * np.sin(ang) + dy * np.cos(ang)
        apothem = side * COS30
        dists = local_x * NORMALS[:, 0] + local_y * NORMALS[:, 1]
        return np.all(dists <= apothem + 1e-9)

    def polygons_disjoint_sat(v1, v2):
        axes = []
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        axes_arr = np.array(axes)
        proj1 = v1 @ axes_arr.T
        proj2 = v2 @ axes_arr.T
        min1 = np.min(proj1, axis=0)
        max1 = np.max(proj1, axis=0)
        min2 = np.min(proj2, axis=0)
        max2 = np.max(proj2, axis=0)
        separated = (max1 < min2 - 1e-9) | (max2 < min1 - 1e-9)
        return np.any(separated)

    def all_inside_outer_fast(inner_positions, outer_center, outer_side, outer_angle):
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices_fast(cx, cy, angle)
            for vx, vy in verts:
                if not point_in_hexagon_fast(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        return True

    def pairwise_disjoint_fast(inner_positions):
        for i in range(len(inner_positions)):
            for j in range(i+1, len(inner_positions)):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                v1 = hexagon_vertices_fast(ci[0], ci[1], ai)
                v2 = hexagon_vertices_fast(cj[0], cj[1], aj)
                if not polygons_disjoint_sat(v1, v2):
                    return False
        return True

    def compute_minimal_outer_fast(inner_positions, outer_angle_deg):
        all_verts = []
        for cx, cy, angle in inner_positions:
            verts = hexagon_vertices_fast(cx, cy, angle)
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        angle_rad = np.deg2rad(outer_angle_deg)
        normal_angles = angle_rad + np.deg2rad(np.arange(30, 360, 60))
        normals_arr = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        dists = all_verts @ normals_arr.T
        max_dists = np.max(dists, axis=0)
        max_apothem_needed = np.max(np.maximum(max_dists, 0))
        return max_apothem_needed / COS30

    def make_hex_cluster():
        positions = []
        positions.append((0.0, 0.0, 0.0))
        for k in range(6):
            angle = np.radians(60 * k)
            x = 2.0 * np.cos(angle)
            y = 2.0 * np.sin(angle)
            positions.append((x, y, 0.0))
        sqrt3 = np.sqrt(3)
        second_ring_candidates = [
            (4.0, 0.0), (3.0, sqrt3), (1.0, 2*sqrt3), (-1.0, 2*sqrt3),
            (-3.0, sqrt3), (-4.0, 0.0), (-3.0, -sqrt3), (-1.0, -2*sqrt3),
            (1.0, -2*sqrt3), (3.0, -sqrt3),
        ]
        best_combo = None
        best_side = float('inf')
        for combo in combinations(range(len(second_ring_candidates)), 5):
            test_positions = positions.copy()
            for idx in combo:
                x, y = second_ring_candidates[idx]
                test_positions.append((x, y, 0.0))
            side = compute_minimal_outer_fast(test_positions, 30.0)
            if side < best_side:
                best_side = side
                best_combo = combo
        for idx in best_combo:
            x, y = second_ring_candidates[idx]
            positions.append((x, y, 0.0))
        return positions, best_side

    def objective(params):
        n_inner = N
        inner_positions = []
        for i in range(n_inner):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_positions.append((x, y, a))
        ox = params[3*n_inner]
        oy = params[3*n_inner+1]
        os = max(params[3*n_inner+2], 0.1)
        oa = params[3*n_inner+3]
        outer_center = (ox, oy)
        outer_side = os
        outer_angle = oa

        inside_ok = all_inside_outer_fast(inner_positions, outer_center, outer_side, outer_angle)
        disjoint_ok = pairwise_disjoint_fast(inner_positions)

        if inside_ok and disjoint_ok:
            return outer_side

        penalty = 0.0
        if not inside_ok:
            for i in range(n_inner):
                cx, cy, angle = inner_positions[i]
                verts = hexagon_vertices_fast(cx, cy, angle)
                for vx, vy in verts:
                    if not point_in_hexagon_fast(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                        penalty += 1.0
        if not disjoint_ok:
            for i in range(n_inner):
                for j in range(i+1, n_inner):
                    ci = (inner_positions[i][0], inner_positions[i][1])
                    cj = (inner_positions[j][0], inner_positions[j][1])
                    ai = inner_positions[i][2]
                    aj = inner_positions[j][2]
                    v1 = hexagon_vertices_fast(ci[0], ci[1], ai)
                    v2 = hexagon_vertices_fast(cj[0], cj[1], aj)
                    if not polygons_disjoint_sat(v1, v2):
                        penalty += 1.0
        return outer_side + penalty * 10.0

    def make_params(inner, oc, os, oa):
        params = []
        for pos in inner:
            params.extend([pos[0], pos[1], pos[2]])
        params.extend([oc[0], oc[1], os, oa])
        return np.array(params, dtype=float)

    # Generate hexagonal cluster initial configuration
    inner_init, initial_outer_side = make_hex_cluster()

    # Find best outer angle for the hex cluster
    best_config = None
    best_side = float('inf')
    for oa in [0.0, 15.0, 30.0, 45.0, 60.0]:
        side = compute_minimal_outer_fast(inner_init, oa)
        if side < best_side:
            best_side = side
            best_config = (inner_init, (0.0, 0.0), side, oa)

    inner_init, oc_init, os_init, oa_init = best_config

    # Ensure validity by expanding if needed
    while not all_inside_outer_fast(inner_init, oc_init, os_init, oa_init):
        os_init += 0.05
        if os_init > 10:
            break

    params_init = make_params(inner_init, oc_init, os_init, oa_init)
    best_params = params_init.copy()
    best_score = objective(params_init)

    # Create additional diverse initial configurations for differential evolution
    initial_configs = [params_init]

    # Config: 4x3 grid
    x_coords = [-3.0, -1.0, 1.0, 3.0]
    y_coords = [-1.8, 0.0, 1.8]
    config1 = []
    for y in y_coords:
        for x in x_coords:
            config1.append((x, y, 0.0))
    os1 = compute_minimal_outer_fast(config1, 30.0)
    while not all_inside_outer_fast(config1, (0.0, 0.0), os1, 30.0):
        os1 += 0.05
        if os1 > 10:
            break
    initial_configs.append(make_params(config1, (0.0, 0.0), os1, 30.0))

    # Config: 2 rows of 6, staggered
    config3 = []
    for i in range(6):
        config3.append((i * 2.0 - 5.0, -0.9, 0.0))
        config3.append((i * 2.0 - 4.0, 0.9, 0.0))
    os3 = compute_minimal_outer_fast(config3, 30.0)
    while not all_inside_outer_fast(config3, (0.0, 0.0), os3, 30.0):
        os3 += 0.05
        if os3 > 10:
            break
    initial_configs.append(make_params(config3, (0.0, 0.0), os3, 30.0))

    # Nelder-Mead refinement from the best initial config
    if time.time() < deadline - 30:
        try:
            res = minimize(objective, params_init, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 0.01, 'fatol': 0.01, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass

    # Differential evolution from multiple starts
    if time.time() < deadline - 15:
        try:
            bounds = [(-5, 5), (-5, 5), (-30, 30)] * N + [(-2, 2), (-2, 2), (3, 10), (-30, 30)]
            # Use the hex cluster as the primary seed
            de_result = differential_evolution(
                objective, bounds,
                seed=42, maxiter=30, popsize=10,
                tol=0.01, polish=False,
                init='sobol'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass

    # Second Nelder-Mead refinement from the best found so far
    if time.time() < deadline - 5 and best_score < 10.0:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 0.005, 'fatol': 0.005, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass

    # Parse final result
    final_params = best_params
    inner = []
    for i in range(N):
        x = final_params[3*i]
        y = final_params[3*i+1]
        a = final_params[3*i+2]
        inner.append((x, y, a))
    oc = (final_params[3*N], final_params[3*N+1])
    os = final_params[3*N+2]
    oa = final_params[3*N+3]

    # Final validation - fallback to initial config if invalid
    if not (all_inside_outer_fast(inner, oc, os, oa) and pairwise_disjoint_fast(inner)):
        inner = inner_init
        oc = oc_init
        os = os_init
        oa = oa_init
        while not all_inside_outer_fast(inner, oc, os, oa):
            os += 0.05
            if os > 10:
                break

    return inner, oc, os, oa
# EVOLVE-BLOCK-END
