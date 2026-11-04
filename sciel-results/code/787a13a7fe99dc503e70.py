import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    sqrt3 = math.sqrt(3)

    # Unit hexagon vertices (pointy-top)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_side_and_penalty(params, outer_center=None, outer_angle=0.0):
        """
        params: [x0..x11, y0..y11, a0..a11] (36 values)
        Returns: (side_length, penalty)
        """
        pos = params[:2*N].reshape(N, 2)
        ang = params[2*N:] # radians

        if outer_center is None:
            outer_center = np.array([0.0, 0.0])

        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang_n = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang_n), math.sin(ang_n)])
        normals = np.array(normals)

        max_proj = 0.0
        polys = []
        for i in range(N):
            cx, cy = pos[i]
            a_rad = ang[i]
            verts = get_hex_verts(cx, cy, 1.0, np.degrees(a_rad))
            polys.append(verts)
            verts_rel = verts - outer_center
            projs = verts_rel @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i

        side = max_proj * 2.0 / sqrt3

        penalty = 0.0
        for i in range(N):
            p1 = polys[i]
            for j in range(i+1, N):
                p2 = polys[j]

                # Quick bounding box check
                min_x1, min_y1 = np.min(p1, axis=0)
                max_x1, max_y1 = np.max(p1, axis=0)
                min_x2, min_y2 = np.min(p2, axis=0)
                max_x2, max_y2 = np.max(p2, axis=0)

                if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                    continue

                # SAT
                min_sep = float('inf')
                axes = []
                for k in range(6):
                    edge = p1[(k+1)%6] - p1[k]
                    n = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-9: axes.append(n/norm)
                for k in range(6):
                    edge = p2[(k+1)%6] - p2[k]
                    n = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(n)
                    if norm > 1e-9: axes.append(n/norm)

                for axis in axes:
                    proj1 = np.dot(p1, axis)
                    proj2 = np.dot(p2, axis)
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)

                    sep = max(min1, min2) - min(max1, max2)
                    if sep < min_sep:
                        min_sep = sep

                if min_sep < 0:
                    penalty += (-min_sep) ** 2 * 100.0

        return side, penalty

    def objective(params):
        side, penalty = calc_side_and_penalty(params)
        return side + penalty

    # Initial configurations
    inits = []

    # 1. Honeycomb lattice (aligned)
    lattice_points = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))

    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_pos = np.array(lattice_points[:12])
    init_pos -= np.mean(init_pos, axis=0)
    init_ang = np.zeros(N)
    inits.append(np.concatenate([init_pos.flatten(), init_ang]))

    # 2. Random starts
    np.random.seed(42)
    for _ in range(10):
        pos = np.random.uniform(-3, 3, (N, 2))
        ang = np.random.uniform(0, 2*math.pi, N)
        inits.append(np.concatenate([pos.flatten(), ang]))

    best_side = float('inf')
    best_params = None

    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 15:
            break

        try:
            res = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )

            final_side, final_penalty = calc_side_and_penalty(res.x)

            if final_penalty < 1e-6:
                if final_side < best_side:
                    best_side = final_side
                    best_params = res.x.copy()

        except Exception:
            pass

    # Polish: Optimize outer center and angle
    if best_params is not None:
        pos = best_params[:2*N].reshape(N, 2)
        ang = best_params[2*N:]

        def outer_obj(outer_params):
            oc = outer_params[:2]
            oa = outer_params[2]
            theta_rad = oa
            normals = []
            for k in range(6):
                ang_n = theta_rad + math.pi/6 + k * math.pi/3
                normals.append([math.cos(ang_n), math.sin(ang_n)])
            normals = np.array(normals)

            max_proj = 0.0
            for i in range(N):
                cx, cy = pos[i]
                a_rad = ang[i]
                verts = get_hex_verts(cx, cy, 1.0, np.degrees(a_rad))
                verts_rel = verts - np.array(oc)
                projs = verts_rel @ normals.T
                max_proj_i = np.max(projs)
                if max_proj_i > max_proj:
                    max_proj = max_proj_i

            return max_proj * 2.0 / sqrt3

        centroid = np.mean(pos, axis=0)
        x0_outer = np.array([centroid[0], centroid[1], 0.0])

        try:
            res_outer = minimize(
                outer_obj,
                x0_outer,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
            )

            if res_outer.fun < best_side:
                best_side = res_outer.fun
                best_outer_center = res_outer.x[:2]
                best_outer_angle = np.degrees(res_outer.x[2])
        except:
            best_outer_center = [0.0, 0.0]
            best_outer_angle = 0.0
    else:
        # Fallback
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        pos = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        ang = np.zeros(N)
        best_params = np.concatenate([pos.flatten(), ang])
        best_side, _ = calc_side_and_penalty(best_params)
        best_outer_center = [0.0, 0.0]
        best_outer_angle = 0.0

    final_pos = best_params[:2*N].reshape(N, 2)
    final_ang = best_params[2*N:]

    inner = []
    for i in range(N):
        inner.append([final_pos[i][0], final_pos[i][1], np.degrees(final_ang[i])])

    return inner, tuple(best_outer_center), best_side, best_outer_angle
