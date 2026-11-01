from scipy.optimize import minimize
from scipy.optimize import differential_evolution, minimize_scalar
import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    N = 12

    def hexagon_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def calc_min_outer_side_fast(all_verts, theta):
        angles = theta + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        widths = dots.max(axis=0) - dots.min(axis=0)
        max_width = np.max(widths)
        apothem = max_width / 2.0
        side = 2 * apothem / np.sqrt(3)
        return side

    def find_best_outer_orientation_fast(all_verts):
        # Use a finer grid search for theta, then a small local polish
        thetas = np.linspace(0, np.pi / 3, 36)
        best_side = np.inf
        best_theta = 0
        for th in thetas:
            s = calc_min_outer_side_fast(all_verts, th)
            if s < best_side:
                best_side = s
                best_theta = th
        # Small local refinement
        if best_side < np.inf:
            res = minimize_scalar(lambda th: calc_min_outer_side_fast(all_verts, th),
                                  bounds=(max(0, best_theta - np.pi / 72), min(np.pi / 3, best_theta + np.pi / 72)),
                                  method='bounded', options={'xatol': 1e-5})
            if res.fun < best_side:
                best_side = res.fun
                best_theta = res.x
        return best_side, best_theta

    def get_overlap_penalty(positions, angles):
        penalty = 0.0
        n = len(positions)
        for i in range(n):
            vi = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            for j in range(i + 1, n):
                vj = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                axes = []
                for v in [vi, vj]:
                    for k in range(len(v)):
                        m = (k + 1) % len(v)
                        edge = v[m] - v[k]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-12:
                            normal /= norm
                            axes.append(normal)
                min_overlap = float('inf')
                for axis in axes:
                    proj1 = np.dot(vi, axis)
                    proj2 = np.dot(vj, axis)
                    min1, max1 = proj1.min(), proj1.max()
                    min2, max2 = proj2.min(), proj2.max()
                    if max1 < min2 or max2 < min1:
                        min_overlap = 0.0
                        break
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap < min_overlap:
                        min_overlap = overlap
                if min_overlap > 0:
                    penalty += min_overlap * min_overlap
        return penalty

    def hexagons_disjoint(positions, angles):
        n = len(positions)
        for i in range(n):
            vi = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            for j in range(i + 1, n):
                vj = hexagon_vertices(positions[j, 0], positions[j, 1], 1.0, angles[j])
                axes = []
                for v in [vi, vj]:
                    for k in range(len(v)):
                        m = (k + 1) % len(v)
                        edge = v[m] - v[k]
                        normal = np.array([-edge[1], edge[0]])
                        norm = np.linalg.norm(normal)
                        if norm > 1e-12:
                            normal /= norm
                            axes.append(normal)
                for axis in axes:
                    proj1 = np.dot(vi, axis)
                    proj2 = np.dot(vj, axis)
                    min1, max1 = proj1.min(), proj1.max()
                    min2, max2 = proj2.min(), proj2.max()
                    if max1 < min2 - 1e-7 or max2 < min1 - 1e-7:
                        break
                else:
                    return False
        return True

    def approx_outer_side(positions, angles):
        all_verts = []
        for i in range(len(positions)):
            v = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        min_side = float('inf')
        for theta in [0, np.pi / 12, np.pi / 6, np.pi / 4, np.pi / 3]:
            side = calc_min_outer_side_fast(all_verts, theta)
            if side < min_side:
                min_side = side
        return min_side

    def full_objective(params):
        if time.time() > deadline:
            return 1e10
        n = len(params) // 3
        positions = np.array([[params[i * 3], params[i * 3 + 1]] for i in range(n)])
        angles = np.array([params[i * 3 + 2] for i in range(n)])
        penalty = get_overlap_penalty(positions, angles)
        if penalty > 0.01:
            return 10.0 + 100.0 * penalty
        side = approx_outer_side(positions, angles)
        return side

    # Generate multiple diverse initial configurations
    def make_initial_config(d, ring1_angles, ring2_angles, ring2_dist_factor):
        positions = []
        angles = []
        positions.append([0.0, 0.0])
        angles.append(0.0)
        for k in range(6):
            ang = np.radians(ring1_angles[k])
            positions.append([d * np.cos(ang), d * np.sin(ang)])
            angles.append(0.0)
        for k in range(5):
            ang = np.radians(ring2_angles[k])
            dist = d * ring2_dist_factor
            positions.append([dist * np.cos(ang), dist * np.sin(ang)])
            angles.append(0.0)
        return np.array(positions), np.array(angles)

    configs = []
    # Config 1: Standard honeycomb
    p1, a1 = make_initial_config(2.02, [0, 60, 120, 180, 240, 300], [30, 90, 150, 210, 270], np.sqrt(3))
    configs.append((p1, a1))
    # Config 2: Slightly larger gap
    p2, a2 = make_initial_config(2.05, [0, 60, 120, 180, 240, 300], [30, 90, 150, 210, 270], np.sqrt(3))
    configs.append((p2, a2))
    # Config 3: Rotated ring 1
    p3, a3 = make_initial_config(2.02, [30, 90, 150, 210, 270, 330], [0, 60, 120, 180, 240], np.sqrt(3))
    configs.append((p3, a3))
    # Config 4: Compact with smaller ring 2 distance
    p4, a4 = make_initial_config(2.02, [0, 60, 120, 180, 240, 300], [30, 90, 150, 210, 270], 2.5)
    configs.append((p4, a4))
    # Config 5: All angles 0, different layout
    p5 = np.array([[0, 0], [2.02, 0], [-2.02, 0], [1.01, 1.75], [-1.01, 1.75], [1.01, -1.75],
                   [-1.01, -1.75], [3.5, 1.75], [3.5, -1.75], [-3.5, 1.75], [-3.5, -1.75], [0, 3.5]])
    a5 = np.zeros(12)
    configs.append((p5, a5))

    # Validate and select initial configs
    valid_configs = []
    for p, a in configs:
        if hexagons_disjoint(p, a):
            valid_configs.append((p, a))

    if not valid_configs:
        # Fallback: very spread out
        p_fb = np.zeros((12, 2))
        for i in range(12):
            ang = 2 * np.pi * i / 12
            p_fb[i] = [5 * np.cos(ang), 5 * np.sin(ang)]
        a_fb = np.zeros(12)
        valid_configs.append((p_fb, a_fb))

    best_valid_side = 1e9
    best_valid_params = None

    bounds = []
    for i in range(12):
        bounds.append((-3.0, 3.0))
        bounds.append((-3.0, 3.0))
        bounds.append((0.0, np.pi / 3))

    def callback(x, convergence):
        nonlocal best_valid_params, best_valid_side
        if time.time() > deadline:
            return True
        pos = np.array([[x[i * 3], x[i * 3 + 1]] for i in range(12)])
        ang = np.array([x[i * 3 + 2] for i in range(12)])
        if hexagons_disjoint(pos, ang):
            all_verts = []
            for i in range(12):
                v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                all_verts.append(v)
            all_verts = np.vstack(all_verts)
            side, _ = find_best_outer_orientation_fast(all_verts)
            if side < best_valid_side:
                best_valid_side = side
                best_valid_params = x.copy()
        return False

    # Run DE from multiple starts
    for idx, (p_init, a_init) in enumerate(valid_configs[:3]):
        if time.time() > deadline - 10:
            break
        init_params = np.concatenate([p_init.flatten(), a_init])
        try:
            result = differential_evolution(
                full_objective,
                bounds=bounds,
                seed=42 + idx,
                maxiter=150,
                popsize=15,
                tol=1e-6,
                polish=False,
                callback=callback,
                init=init_params
            )
            if result.fun < 10.0:
                pos = np.array([[result.x[i * 3], result.x[i * 3 + 1]] for i in range(12)])
                ang = np.array([result.x[i * 3 + 2] for i in range(12)])
                if hexagons_disjoint(pos, ang):
                    all_verts = []
                    for i in range(12):
                        v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                        all_verts.append(v)
                    all_verts = np.vstack(all_verts)
                    side, _ = find_best_outer_orientation_fast(all_verts)
                    if side < best_valid_side:
                        best_valid_side = side
                        best_valid_params = result.x.copy()
        except Exception as e:
            pass

    # Final polish with local optimization if time permits
    if best_valid_params is not None and time.time() < deadline - 5:
        pos = np.array([[best_valid_params[i * 3], best_valid_params[i * 3 + 1]] for i in range(12)])
        ang = np.array([best_valid_params[i * 3 + 2] for i in range(12)])

        def local_obj(params):
            if time.time() > deadline:
                return 1e10
            p = np.array([[params[i * 3], params[i * 3 + 1]] for i in range(12)])
            a = np.array([params[i * 3 + 2] for i in range(12)])
            penalty = get_overlap_penalty(p, a)
            if penalty > 0.001:
                return 10.0 + 1000.0 * penalty
            return approx_outer_side(p, a)

        try:
            res = minimize(local_obj, best_valid_params, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5})
            if res.fun < 10.0:
                pos = np.array([[res.x[i * 3], res.x[i * 3 + 1]] for i in range(12)])
                ang = np.array([res.x[i * 3 + 2] for i in range(12)])
                if hexagons_disjoint(pos, ang):
                    all_verts = []
                    for i in range(12):
                        v = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, ang[i])
                        all_verts.append(v)
                    all_verts = np.vstack(all_verts)
                    side, _ = find_best_outer_orientation_fast(all_verts)
                    if side < best_valid_side:
                        best_valid_side = side
                        best_valid_params = res.x.copy()
        except Exception as e:
            pass

    if best_valid_side < 1e8 and best_valid_params is not None:
        best_positions = np.array([[best_valid_params[i * 3], best_valid_params[i * 3 + 1]] for i in range(12)])
        best_angles = np.array([best_valid_params[i * 3 + 2] for i in range(12)])
        all_verts = []
        for i in range(12):
            v = hexagon_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
            all_verts.append(v)
        all_verts = np.vstack(all_verts)
        outer_side, outer_angle_rad = find_best_outer_orientation_fast(all_verts)
        angles = outer_angle_rad + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        mins = dots.min(axis=0)
        maxs = dots.max(axis=0)
        mids = (mins + maxs) / 2.0
        c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
        outer_center = tuple(c)
        outer_angle_deg = np.degrees(outer_angle_rad)
    else:
        outer_center = (0.0, 0.0)
        outer_side = 10.0
        outer_angle_deg = 0.0
        best_positions = valid_configs[0][0]
        best_angles = valid_configs[0][1]

    inner = [[best_positions[i, 0], best_positions[i, 1], np.degrees(best_angles[i])] for i in range(12)]
    return inner, outer_center, outer_side, outer_angle_deg
# EVOLVE-BLOCK-END
