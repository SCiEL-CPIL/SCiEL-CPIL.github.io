import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    sqrt3 = math.sqrt(3.0)

    # Unit hexagon vertices at angle 0 (flat-top: vertices at 0,60,...,300 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """Exact minimum side length of a regular hexagon (center 0,0) containing all inner hexagons."""
        theta_rad = math.radians(outer_angle)
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k * math.pi/3),
                             math.sin(theta_rad + math.pi/6 + k * math.pi/3)] for k in range(6)])
        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            mp = np.max(projs)
            if mp > max_proj:
                max_proj = mp
        return max_proj * 2.0 / sqrt3

    def calc_min_outer_side_smooth(positions, angles, outer_angle=0.0, k=60.0):
        """Smooth (log-sum-exp) surrogate of the max projection -> smooth outer side length."""
        theta_rad = math.radians(outer_angle)
        normals = np.array([[math.cos(theta_rad + math.pi/6 + k2 * math.pi/3),
                             math.sin(theta_rad + math.pi/6 + k2 * math.pi/3)] for k2 in range(6)])
        all_projs = []
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            all_projs.append(verts @ normals.T)
        all_projs = np.concatenate(all_projs)
        vmax = np.max(all_projs)
        # log-sum-exp of all projections
        shifted = all_projs - vmax
        lse = vmax + math.log(np.sum(np.exp(k * shifted)) / k) / k
        return lse * 2.0 / sqrt3

    def sat_gap(poly1, poly2):
        """Maximum SAT separation. Positive = disjoint (gap), negative = overlap depth."""
        max_gap = -float('inf')
        for poly in (poly1, poly2):
            for i in range(6):
                edge = poly[(i + 1) % 6] - poly[i]
                axis = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(axis)
                if norm < 1e-9:
                    continue
                axis = axis / norm
                p1 = poly1 @ axis
                p2 = poly2 @ axis
                min1, max1 = np.min(p1), np.max(p1)
                min2, max2 = np.min(p2), np.max(p2)
                if max1 < min2:
                    gap = min2 - max1
                elif max2 < min1:
                    gap = min1 - max2
                else:
                    gap = -(min(max1, max2) - max(min1, min2))
                if gap > max_gap:
                    max_gap = gap
        return max_gap

    def check_disjoint_fast(poly1, poly2):
        """True if disjoint (or touching)."""
        return sat_gap(poly1, poly2) >= -1e-9

    def is_valid(positions, angles):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i + 1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    return False
        return True

    # ---- Objective: smooth outer side + smooth continuous overlap penalty ----
    def obj_general(params, k=60.0):
        if time.time() > deadline - 5:
            return 1000.0
        positions = params[:2 * N].reshape(N, 2)
        angles = params[2 * N:]

        side_len = calc_min_outer_side_smooth(positions, angles, 0.0, k)

        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0.0:
                    # quadratic growth with overlap depth -> smooth, informative
                    penalty += (-gap) ** 2
        return side_len + 10.0 * penalty

    # ---- Initial guess: honeycomb subset, centered ----
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
    lattice_points.sort(key=lambda p: p[0] ** 2 + p[1] ** 2)
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    x0 = np.concatenate([init_positions.flatten(), init_angles])

    best_score = float('inf')
    best_result = None

    def try_optimize(x_start, maxiter, xatol, fatol, k):
        nonlocal best_score, best_result
        try:
            res = minimize(
                obj_general,
                x_start,
                args=(k,),
                method='Nelder-Mead',
                options={'maxiter': maxiter, 'xatol': xatol, 'fatol': fatol, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass

    # Primary run: larger smoothness (k) for a well-behaved landscape
    if time.time() < deadline - 15:
        try_optimize(x0, 4000, 1e-6, 1e-9, 60.0)

    # Restart with random perturbations to escape local minima
    if time.time() < deadline - 15:
        for restart in range(8):
            if time.time() > deadline - 15:
                break
            x0_perturbed = x0.copy()
            x0_perturbed += np.random.normal(0, 0.4, size=x0.shape)
            try_optimize(x0_perturbed, 1200, 1e-5, 1e-7, 40.0)

    # ---- High-precision polish pass on the best candidate ----
    if best_result is not None and time.time() < deadline - 8:
        # Tighten with a very smooth objective and tighter tolerances
        try:
            res = minimize(
                obj_general,
                best_result,
                args=(120.0,),
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass

    # ---- Finalize ----
    if best_result is not None:
        final_positions = best_result[:2 * N].reshape(N, 2)
        final_angles = best_result[2 * N:]

        # Verify validity; if invalid, fall back to a safe configuration
        if not is_valid(final_positions, final_angles):
            final_positions = np.array([
                [0, 0], [2.1, 0], [-2.1, 0],
                [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
                [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
                [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
            ])
            final_angles = np.zeros(N)

        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    else:
        final_positions = np.array([
            [0, 0], [2.1, 0], [-2.1, 0],
            [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
            [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
            [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
        ])
        final_angles = np.zeros(N)
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])

    return inner, (0.0, 0.0), final_side, 0.0
