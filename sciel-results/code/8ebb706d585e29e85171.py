import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    sqrt3 = math.sqrt(3)

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        base = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
        verts = base @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def hex_edges(verts):
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([-edges[:, 1], edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normals = normals / norms
        return edges, normals

    def sat_min_separation(verts_a, verts_b):
        edges_a, normals_a = hex_edges(verts_a)
        edges_b, normals_b = hex_edges(verts_b)
        all_normals = np.vstack([normals_a, normals_b])
        min_overlap = float('inf')
        best_axis = None
        for n in all_normals:
            proj_a = verts_a @ n
            proj_b = verts_b @ n
            min_a, max_a = np.min(proj_a), np.max(proj_a)
            min_b, max_b = np.min(proj_b), np.max(proj_b)
            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap < 0:
                return 0.0, n
            if overlap < min_overlap:
                min_overlap = overlap
                best_axis = n
        if best_axis is None:
            return 0.0, np.array([1.0, 0.0])
        center_a = np.mean(verts_a, axis=0)
        center_b = np.mean(verts_b, axis=0)
        diff = center_a - center_b
        if np.dot(diff, best_axis) < 0:
            best_axis = -best_axis
        return min_overlap, best_axis

    def calc_outer_side(positions, angles, outer_angle=0.0):
        theta_rad = math.radians(outer_angle)
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3),
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] for k in range(6)])
        max_proj = 0.0
        for i in range(N):
            verts = get_hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i])
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        return max_proj * 2.0 / sqrt3

    def check_disjoint(positions, angles, tol=1e-5):
        for i in range(N):
            for j in range(i+1, N):
                vi = get_hex_verts(positions[i,0], positions[i,1], 1.0, angles[i])
                vj = get_hex_verts(positions[j,0], positions[j,1], 1.0, angles[j])
                sep, _ = sat_min_separation(vi, vj)
                if sep > tol:
                    return False
        return True

    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        return calc_outer_side(positions, angles, 0.0)

    def overlap_constraint(params, i, j):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        verts_i = get_hex_verts(positions[i, 0], positions[i, 1], 1.0, angles[i])
        verts_j = get_hex_verts(positions[j, 0], positions[j, 1], 1.0, angles[j])
        sep, axis = sat_min_separation(verts_i, verts_j)
        return -sep

    # --- Constructive initial placements ---
    v1 = np.array([sqrt3, 0.0])
    v2 = np.array([sqrt3/2.0, 1.5])

    def lattice_positions(ring2_count=5):
        positions = [np.array([0.0, 0.0])]
        for k in range(6):
            ang = k * math.pi / 3
            pos = sqrt3 * np.array([math.cos(ang), math.sin(ang)])
            positions.append(pos)
        ring2_pts = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                d = max(abs(i), abs(j), abs(i+j))
                if d == 2:
                    pos = i * v1 + j * v2
                    ring2_pts.append(pos)
        ring2_sorted = sorted(ring2_pts, key=lambda p: math.atan2(p[1], p[0]))
        for k in range(ring2_count):
            positions.append(ring2_sorted[k])
        return np.array(positions)

    # Config 1: 1+6+5 hexagonal cluster
    pos1 = lattice_positions(5)
    # Config 2: 3x4 grid
    pos2 = []
    for row in range(3):
        for col in range(4):
            offset = 0.5 * v1 if row % 2 == 1 else 0
            pos = col * v1 + row * v2 + offset
            pos2.append(pos)
    pos2 = np.array(pos2)
    # Config 3: 4x3 grid
    pos3 = []
    for row in range(4):
        for col in range(3):
            offset = 0.5 * v1 if row % 2 == 1 else 0
            pos = col * v1 + row * v2 + offset
            pos3.append(pos)
    pos3 = np.array(pos3)
    # Config 4: 1+6+5 different arc
    pos4 = [np.array([0.0, 0.0])]
    for k in range(6):
        ang = k * math.pi / 3
        pos4.append(sqrt3 * np.array([math.cos(ang), math.sin(ang)]))
    ring2_pts = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            d = max(abs(i), abs(j), abs(i+j))
            if d == 2:
                pos = i * v1 + j * v2
                ring2_pts.append(pos)
    ring2_sorted2 = sorted(ring2_pts, key=lambda p: math.atan2(p[1], p[0]))
    for k in range(5, 10):
        pos4.append(ring2_sorted2[k % len(ring2_sorted2)])
    pos4 = np.array(pos4)

    initial_configs = [pos1, pos2, pos3, pos4]

    best_result = None
    best_score = float('inf')

    # --- Phase 1: SLSQP with fixed angles (fast, deterministic) ---
    for cfg_idx, init_pos in enumerate(initial_configs):
        if time.time() > deadline - 40:
            break
        center = np.mean(init_pos, axis=0)
        init_pos = init_pos - center
        x0 = np.concatenate([init_pos.flatten(), np.zeros(N)])
        constraints = []
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({'type': 'ineq', 'fun': lambda params, i=i, j=j: overlap_constraint(params, i, j)})
        bounds = [(-6, 6), (-6, 6)] * N + [(0, 0)] * N
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 400, 'ftol': 1e-10})
            if res.fun < best_score:
                final_pos = res.x[:2*N].reshape(N, 2)
                final_ang = res.x[2*N:]
                if check_disjoint(final_pos, final_ang, 1e-5):
                    side = calc_outer_side(final_pos, final_ang, 0.0)
                    if side < best_score:
                        best_score = side
                        best_result = res.x.copy()
        except Exception:
            pass

    # --- Phase 2: Differential Evolution with free angles (global search) ---
    if time.time() < deadline - 30:
        try:
            bounds_de = [(-5, 5), (-5, 5)] * N + [(-30, 30)] * N

            def de_objective(params):
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:]
                side = calc_outer_side(positions, angles, 0.0)
                penalty = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        vi = get_hex_verts(positions[i,0], positions[i,1], 1.0, angles[i])
                        vj = get_hex_verts(positions[j,0], positions[j,1], 1.0, angles[j])
                        sep, _ = sat_min_separation(vi, vj)
                        if sep > 0:
                            penalty += sep ** 2
                return side + 50.0 * penalty

            # Use multiple starting points
            for cfg_idx in range(min(2, len(initial_configs))):
                if time.time() > deadline - 25:
                    break
                init_pos = initial_configs[cfg_idx]
                center = np.mean(init_pos, axis=0)
                init_pos = init_pos - center
                x0 = np.concatenate([init_pos.flatten(), np.zeros(N)])
                bounds_de_local = [(x0[k]-2, x0[k]+2) for k in range(2*N)] + [(-30, 30)] * N

                de_res = differential_evolution(
                    de_objective, bounds_de_local,
                    seed=42+cfg_idx, maxiter=30, tol=1e-8,
                    mutation=(0.3, 0.9), recombination=0.7,
                    popsize=15, polish=False,
                    updating='immediate'
                )
                if de_res.fun < best_score + 1:
                    final_pos = de_res.x[:2*N].reshape(N, 2)
                    final_ang = de_res.x[2*N:]
                    if check_disjoint(final_pos, final_ang, 1e-5):
                        side = calc_outer_side(final_pos, final_ang, 0.0)
                        if side < best_score:
                            best_score = side
                            best_result = de_res.x.copy()
        except Exception:
            pass

    # --- Phase 3: SLSQP polish with free angles from best DE result ---
    if best_result is not None and time.time() < deadline - 15:
        try:
            constraints = []
            for i in range(N):
                for j in range(i+1, N):
                    constraints.append({'type': 'ineq', 'fun': lambda params, i=i, j=j: overlap_constraint(params, i, j)})
            bounds = [(-6, 6), (-6, 6)] * N + [(-30, 30)] * N
            res = minimize(objective, best_result, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': 500, 'ftol': 1e-10})
            if res.fun < best_score:
                final_pos = res.x[:2*N].reshape(N, 2)
                final_ang = res.x[2*N:]
                if check_disjoint(final_pos, final_ang, 1e-5):
                    side = calc_outer_side(final_pos, final_ang, 0.0)
                    if side < best_score:
                        best_score = side
                        best_result = res.x.copy()
        except Exception:
            pass

    # --- Phase 4: Nelder-Mead final polish ---
    if best_result is not None and time.time() < deadline - 8:
        try:
            def penalized_objective(params):
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:]
                side = calc_outer_side(positions, angles, 0.0)
                penalty = 0.0
                for i in range(N):
                    for j in range(i+1, N):
                        vi = get_hex_verts(positions[i,0], positions[i,1], 1.0, angles[i])
                        vj = get_hex_verts(positions[j,0], positions[j,1], 1.0, angles[j])
                        sep, _ = sat_min_separation(vi, vj)
                        if sep > 0:
                            penalty += sep ** 2
                return side + 100.0 * penalty

            res_polish = minimize(penalized_objective, best_result, method='Nelder-Mead',
                                  options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-9})
            if res_polish.fun < best_score + 1e-6:
                final_pos = res_polish.x[:2*N].reshape(N, 2)
                final_ang = res_polish.x[2*N:]
                if check_disjoint(final_pos, final_ang, 1e-5):
                    side = calc_outer_side(final_pos, final_ang, 0.0)
                    if side < best_score:
                        best_score = side
                        best_result = res_polish.x.copy()
        except Exception:
            pass

    # --- Fallback ---
    if best_result is None or best_score > 10:
        pos_fb = lattice_positions(5)
        center = np.mean(pos_fb, axis=0)
        pos_fb = pos_fb - center
        angles = np.zeros(N)
        best_score = calc_outer_side(pos_fb, angles, 0.0)
        best_result = np.concatenate([pos_fb.flatten(), angles])

    positions = best_result[:2*N].reshape(N, 2)
    angles = best_result[2*N:]
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles[i])])
    final_side = calc_outer_side(positions, angles, 0.0)
    return inner, (0.0, 0.0), float(final_side), 0.0
