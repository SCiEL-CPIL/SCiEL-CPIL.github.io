import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 12
SQRT3 = np.sqrt(3.0)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def outer_hex_side_length(vertices, outer_angle_rad):
        """
        Compute the minimum side length S of a regular hexagon with given center (0,0)
        and rotation outer_angle_rad that contains all vertices.
        
        For a regular hexagon centered at origin with rotation phi, the boundary in
        direction theta (angle from x-axis) is at distance:
        r(theta) = S * cos(pi/6) / cos((theta - phi) mod (2pi/3) - pi/3)
        
        Wait, let me use the support function approach.
        The hexagon is the intersection of 6 half-planes. For a hexagon with side S
        and rotation phi, the 6 side normals are at angles phi + k*pi/3 for k=0..5.
        The distance from center to each side (apothem) is a = S * sqrt(3)/2.
        
        A point p is inside iff for all k=0..5:
            p . n_k <= a
        where n_k = (cos(phi + k*pi/3), sin(phi + k*pi/3))
        
        So S >= 2/sqrt(3) * max_k (p . n_k) for all vertices p.
        Actually: p . n_k <= a = S * sqrt(3)/2
        So S >= 2 * (p . n_k) / sqrt(3)
        And S >= 2 * max_k (p . n_k) / sqrt(3) for each p.
        Overall: S >= 2/sqrt(3) * max_p max_k (p . n_k)
        """
        normals = np.column_stack([
            np.cos(outer_angle_rad + np.pi / 3.0 * np.arange(6)),
            np.sin(outer_angle_rad + np.pi / 3.0 * np.arange(6))
        ])
        # vertices: (M, 2), normals: (6, 2)
        # projections: (M, 6)
        projections = vertices @ normals.T
        max_proj = np.max(projections)
        return 2.0 * max_proj / SQRT3

    def compute_score(params):
        """
        params: array of shape (N, 3) where each row is (x, y, angle_rad)
        Returns (score, penalty) where score is outer hex side length,
        penalty is 0 if valid (all disjoint and contained).
        """
        if time.time() > deadline:
            return 1e10, 1e10

        positions = params[:, :2]
        angles = params[:, 2]

        # Compute all vertices
        all_vertices = []
        for i in range(N):
            x, y = positions[i]
            a = angles[i]
            verts = hexagon_vertices(x, y, 1.0, a)
            all_vertices.append(verts)

        # Check pairwise disjointness using SAT
        for i in range(N):
            for j in range(i + 1, N):
                p1 = all_vertices[i]
                p2 = all_vertices[j]
                # Quick bbox check
                if not _bbox_overlap(p1, p2, eps=1e-6):
                    continue
                if not _sat_disjoint(p1, p2, eps=1e-6):
                    return 1e10, 1e10

        # Compute outer hexagon side length - try several rotations
        # Combine all vertices
        all_verts = np.vstack(all_vertices)

        # Try optimizing the outer hexagon rotation
        best_side = 1e10
        # Try a grid of rotations plus local optimization
        for rot in np.linspace(0, np.pi / 3, 7):  # 7 angles over 60 degrees (hex symmetry)
            s = outer_hex_side_length(all_verts, rot)
            if s < best_side:
                best_side = s

        # Local polish on rotation
        if best_side < 1e9:
            def rot_obj(r):
                return outer_hex_side_length(all_verts, r[0])
            try:
                res = minimize(rot_obj, [best_rot_init], method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
                if res.fun < best_side:
                    best_side = res.fun
            except:
                pass

        if best_side > 1e9:
            return 1e10, 1e10
        return best_side, 0.0

    def _bbox_overlap(p1, p2, eps=1e-6):
        """Quick bounding box overlap check."""
        min1 = p1.min(axis=0) - eps
        max1 = p1.max(axis=0) + eps
        min2 = p2.min(axis=0) - eps
        max2 = p2.max(axis=0) + eps
        if max1[0] < min2[0] or max2[0] < min1[0]:
            return False
        if max1[1] < min2[1] or max2[1] < min1[1]:
            return False
        return True

    def _sat_disjoint(p1, p2, eps=1e-6):
        """SAT: return True if polygons are disjoint."""
        for poly in [p1, p2]:
            for i in range(len(poly)):
                j = (i + 1) % len(poly)
                edge = poly[j] - poly[i]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-12:
                    continue
                normal /= norm
                proj1 = p1 @ normal
                proj2 = p2 @ normal
                if proj1.max() < proj2.min() - eps or proj2.max() < proj1.min() - eps:
                    return True
        return False

    # Initial honeycomb configuration
    coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
    ]
    init_positions = []
    for q, r in coords:
        x = SQRT3 * (q + r / 2.0)
        y = 1.5 * r
        init_positions.append([x, y, 0.0])
    init_params = np.array(init_positions)

    # Verify initial is valid
    init_score, init_penalty = compute_score(init_params)
    if init_penalty > 0:
        # Spread them out
        init_params = np.zeros((N, 3))
        for i in range(N):
            angle = 2 * np.pi * i / N
            init_params[i, 0] = 2.5 * np.cos(angle)
            init_params[i, 1] = 2.5 * np.sin(angle)
            init_params[i, 2] = 0.0
        init_score, init_penalty = compute_score(init_params)

    best_params = init_params.copy()
    best_score = init_score
    best_penalty = init_penalty

    # Simulated Annealing with force-based model
    rng = np.random.RandomState(42)
    T = 2.0
    T_min = 0.001
    cooling_rate = 0.995
    max_iters = 200000

    current_params = init_params.copy()
    current_score, current_penalty = compute_score(current_params)
    current_total = current_score + current_penalty

    iter_count = 0
    while iter_count < max_iters and time.time() < deadline - 5.0:
        iter_count += 1

        # Pick a random hexagon to perturb
        idx = rng.randint(N)
        new_params = current_params.copy()

        # Perturb position and angle
        scale = T * 0.3
        new_params[idx, 0] += rng.normal(0, scale)
        new_params[idx, 1] += rng.normal(0, scale)
        new_params[idx, 2] += rng.normal(0, scale * 0.1)
        # Wrap angle
        new_params[idx, 2] = (new_params[idx, 2] + np.pi) % (2 * np.pi) - np.pi

        new_score, new_penalty = compute_score(new_params)
        new_total = new_score + new_penalty

        delta = new_total - current_total

        if delta < 0 or rng.random() < np.exp(-delta / max(T, T_min)):
            current_params = new_params
            current_score = new_score
            current_penalty = new_penalty
            current_total = new_total

            if current_total < best_score + best_penalty:
                best_params = current_params.copy()
                best_score = current_score
                best_penalty = current_penalty

        T *= cooling_rate
        if T < T_min:
            T = T_min

    # If SA didn't improve or found invalid, try differential evolution on a subset
    if time.time() < deadline - 15.0:
        # Try DE with smooth objective approximation
        def de_objective(flat_params):
            if time.time() > deadline:
                return 1e10
            params = flat_params.reshape((N, 3))
            score, penalty = compute_score(params)
            return score + penalty

        bounds = [(-5, 5)] * (2 * N) + [(-np.pi, np.pi)] * N
        # Use init as starting point via x0
        try:
            result = differential_evolution(
                de_objective,
                bounds,
                seed=42,
                maxiter=50,
                popsize=10,
                tol=1e-3,
                polish=True,
                init='sobol'
            )
            if result.fun < best_score + best_penalty and result.fun < 1e9:
                de_params = result.x.reshape((N, 3))
                de_score, de_penalty = compute_score(de_params)
                if de_penalty == 0:
                    best_params = de_params
                    best_score = de_score
                    best_penalty = de_penalty
        except Exception:
            pass

    # Final: if best is invalid, fall back to init
    if best_penalty > 0:
        best_params = init_params
        best_score = init_score
        best_penalty = init_penalty

    # Compute final outer hexagon rotation optimally
    all_vertices = []
    for i in range(N):
        x, y = best_params[i, :2]
        a = best_params[i, 2]
        verts = hexagon_vertices(x, y, 1.0, a)
        all_vertices.append(verts)
    all_verts = np.vstack(all_vertices)

    # Optimize outer rotation
    def rot_obj(r):
        return outer_hex_side_length(all_verts, r[0])

    best_rot = 0.0
    best_side = 1e10
    for rot in np.linspace(0, np.pi / 3, 13):
        s = outer_hex_side_length(all_verts, rot)
        if s < best_side:
            best_side = s
            best_rot = rot

    try:
        res = minimize(rot_obj, [best_rot], method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        if res.fun < best_side:
            best_side = res.fun
            best_rot = res.x[0]
    except:
        pass

    # Construct inner list
    inner = []
    for i in range(N):
        x = float(best_params[i, 0])
        y = float(best_params[i, 1])
        ang_deg = float(np.degrees(best_params[i, 2]))
        inner.append([x, y, ang_deg])

    # Outer hexagon: center (0,0), side best_side, rotation best_rot (in degrees)
    return inner, (0.0, 0.0), float(best_side), float(np.degrees(best_rot))

# EVOLVE-BLOCK-END
