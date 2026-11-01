"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize, linprog, minimize_scalar

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
DEADLINE = 145.0  # safety margin before 150s kill
START_TIME = time.time()


def logsumexp(a, b=1.0):
    """Stable log-sum-exp for array a with temperature b."""
    a = np.asarray(a, dtype=np.float64)
    amax = np.max(a)
    if not np.isfinite(amax):
        return amax
    return amax + math.log(np.sum(np.exp((a - amax) / b))) * b


def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts


def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    max_projection = 0.0

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    return 2.0 * max_projection / SQRT3


def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []

    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)

        if max1 < min2 + eps or max2 < min1 + eps:
            return False

    return True


def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True


def check_inside_outer(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    apothem = outer_side * SQRT3 / 2.0

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                if lx * nx + ly * ny > apothem + eps:
                    return False
    return True


def axial_to_cartesian(axial_list):
    """Convert axial coordinates to cartesian coordinates for unit hexagons."""
    pts = []
    for q, r in axial_list:
        x = 1.5 * q
        y = SQRT3 * (r + q / 2.0)
        pts.append((x, y))
    return np.array(pts)


def calc_S_for_beta(pts, beta):
    """
    Compute minimum side length S for a set of points with outer hexagon rotation beta.
    Uses LP to find optimal center and apothem.
    """
    if time.time() - START_TIME > DEADLINE:
        return float('inf'), None, None

    normals = []
    for k in range(6):
        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(ang), math.sin(ang)))

    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []

    for nx, ny in normals:
        for px, py in pts:
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(px * nx + py * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(px * nx + py * ny)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0, None)]

    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            D = res.x[2]
            cx = res.x[0]
            cy = res.x[1]
            S = (D + APOTHEM) / APOTHEM
            return S, (cx, cy), D
        else:
            return float('inf'), None, None
    except Exception:
        return float('inf'), None, None


def optimize_outer_for_points(pts):
    """
    Optimize outer hexagon rotation and center for given point set.
    Returns (best_S, best_beta, best_center).
    """
    best_S = float('inf')
    best_beta = 0.0
    best_center = (0.0, 0.0)

    if time.time() - START_TIME > DEADLINE - 5:
        return float('inf'), 0.0, (0.0, 0.0)

    # Coarse grid search over beta
    for i in range(37):
        beta = i * (math.pi / 6) / 36
        S, center, D = calc_S_for_beta(pts, beta)
        if S < best_S:
            best_S = S
            best_beta = beta
            best_center = center

    # Fine search using minimize_scalar
    try:
        if time.time() - START_TIME < DEADLINE - 5:
            res = minimize_scalar(
                lambda b: calc_S_for_beta(pts, b)[0],
                bounds=(0, math.pi / 6),
                method='bounded',
                options={'maxiter': 200, 'xatol': 1e-10}
            )
            if res.fun < best_S:
                best_S = res.fun
                best_beta = res.x
                _, best_center, _ = calc_S_for_beta(pts, best_beta)
    except Exception:
        pass

    return best_S, best_beta, best_center


def force_directed_compression(initial_positions, n_iter=500, dt=0.001):
    """
    Force-directed algorithm to compress hexagons into densest configuration.
    Returns optimized positions and rotations.
    """
    positions = np.array(initial_positions, dtype=np.float64).copy()
    rotations = np.zeros(N, dtype=np.float64)

    # Minimum center-to-center distance for non-overlapping hexagons
    min_dist = SQRT3 - 1e-6

    for iteration in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 10:
            break

        forces = np.zeros_like(positions)
        torques = np.zeros(N, dtype=np.float64)

        # Repulsive forces between overlapping hexagons
        for i in range(N):
            for j in range(i + 1, N):
                dx = positions[j][0] - positions[i][0]
                dy = positions[j][1] - positions[i][1]
                dist = math.sqrt(dx * dx + dy * dy)

                if dist < min_dist and dist > 1e-9:
                    # Repulsive force proportional to overlap
                    overlap = min_dist - dist
                    force_mag = 10.0 * overlap
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i] += [-fx, -fy]
                    forces[j] += [fx, fy]

                    # Torque to align hexagons for better packing
                    # Rotate towards alignment that minimizes overlap
                    angle_diff = rotations[j] - rotations[i]
                    torque_mag = 0.1 * overlap
                    torques[i] += torque_mag * math.sin(angle_diff)
                    torques[j] -= torque_mag * math.sin(angle_diff)

        # Attractive force towards center of mass
        com = np.mean(positions, axis=0)
        for i in range(N):
            dx = positions[i][0] - com[0]
            dy = positions[i][1] - com[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 1e-9:
                force_mag = 0.01 * dist
                forces[i] += [-force_mag * dx / dist, -force_mag * dy / dist]

        # Update positions and rotations
        for i in range(N):
            positions[i] += forces[i] * dt
            rotations[i] += torques[i] * dt

    return positions, rotations


def construct():
    start_time = time.time()

    # Multiple candidate axial coordinate sets
    candidate_axials = [
        # Classic hexagonal cluster
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (1, 1), (0, 2), (-1, 2)],
        # Star pattern
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 0), (-2, 0), (0, 2), (0, -2)],
        # Linear extension
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (0, 1), (1, 1), (2, 1), (3, 1),
         (0, 2), (1, 2), (2, 2)],
        # Triangular
        [(0, 0), (1, 0), (2, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (-1, 1), (3, 1)],
        # Compact diamond
        [(0, 0), (1, 0),
         (0, 1), (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (0, 3), (1, 3), (2, 3)],
        # Asymmetric
        [(0, 0), (1, 0), (2, 0),
         (1, 1), (2, 1),
         (0, 2), (1, 2), (2, 2),
         (-1, 1), (3, 1), (0, -1)],
        # Dense cluster variant
        [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
         (2, 1), (1, 2), (-1, 2), (0, 3)],
        # Elongated
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
         (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
        # Mixed
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (0, 1), (1, 1),
         (0, 2), (1, 2),
         (0, 3), (1, 3),
         (0, 4)],
        # Another variant
        [(0, 0), (1, 0), (2, 0), (3, 0),
         (0, 1), (1, 1), (2, 1),
         (1, 2), (2, 2), (3, 2)],
    ]

    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Phase 1: Evaluate each candidate with LP-based outer optimization
    for idx, axial in enumerate(candidate_axials):
        if time.time() - start_time > DEADLINE - 20:
            break

        pts = axial_to_cartesian(axial)

        # Optimize outer hexagon
        S, beta, center = optimize_outer_for_points(pts)

        if S < best_side:
            best_side = S
            best_outer_angle = math.degrees(beta)
            best_outer_center = center

            # Build inner list
            inner_list = []
            for i, (q, r) in enumerate(axial):
                x = pts[i][0]
                y = pts[i][1]
                inner_list.append([x, y, 0.0])
            best_inner = inner_list

    # Phase 2: Force-directed compression on best candidate
    if best_inner is not None and time.time() - start_time < DEADLINE - 15:
        initial_positions = [(x, y) for x, y, _ in best_inner]
        compressed_positions, compressed_rotations = force_directed_compression(
            initial_positions, n_iter=300, dt=0.0005
        )

        # Check if compressed configuration is valid
        compressed_inner = []
        for i in range(N):
            compressed_inner.append([
                compressed_positions[i][0],
                compressed_positions[i][1],
                math.degrees(compressed_rotations[i])
            ])

        if check_all_disjoint(compressed_inner):
            # Optimize outer for compressed configuration
            pts_compressed = np.array(compressed_positions)
            S_comp, beta_comp, center_comp = optimize_outer_for_points(pts_compressed)

            if S_comp < best_side:
                best_side = S_comp
                best_outer_angle = math.degrees(beta_comp)
                best_outer_center = center_comp
                best_inner = compressed_inner

    # Phase 3: Joint optimization with smooth objective
    if best_inner is not None and time.time() - start_time < DEADLINE - 10:
        def compute_required_side_smooth(inner_list, outer_center, outer_angle_deg, temperature=0.01):
            if not inner_list:
                return 0.0

            ox, oy = outer_center
            normals = []
            for i in range(6):
                theta = math.radians(30 + i * 60)
                normals.append((math.cos(theta), math.sin(theta)))

            cos_a = math.cos(math.radians(outer_angle_deg))
            sin_a = math.sin(math.radians(outer_angle_deg))

            all_projections = []
            for (ix, iy, iangle) in inner_list:
                verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
                for v in verts:
                    dx = v[0] - ox
                    dy = v[1] - oy
                    lx = dx * cos_a + dy * sin_a
                    ly = -dx * sin_a + dy * cos_a

                    for nx, ny in normals:
                        proj = lx * nx + ly * ny
                        all_projections.append(proj)

            if len(all_projections) == 0:
                return 0.0

            proj_array = np.array(all_projections, dtype=np.float64)
            smooth_max = logsumexp(proj_array, temperature)
            return 2.0 * smooth_max / SQRT3

        def objective_joint_smooth(params, temperature=0.01):
            inner_list = []
            for i in range(N):
                x = params[i * 3]
                y = params[i * 3 + 1]
                a = params[i * 3 + 2]
                inner_list.append([x, y, a])

            cx = params[N * 3]
            cy = params[N * 3 + 1]
            angle = params[N * 3 + 2]

            side = compute_required_side_smooth(inner_list, (cx, cy), angle, temperature)

            # Penalty for overlaps
            penalty = 0.0
            min_dist = SQRT3 - 0.02
            for i in range(N):
                for j in range(i + 1, N):
                    xi, yi, _ = inner_list[i]
                    xj, yj, _ = inner_list[j]
                    dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                    if dist < min_dist:
                        penalty += (min_dist - dist) ** 2

            return side + 10.0 * penalty

        init_params = []
        for x, y, a in best_inner:
            init_params.extend([x, y, a])
        init_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

        best_joint_side = best_side
        best_joint_inner = [list(x) for x in best_inner]
        best_joint_center = best_outer_center
        best_joint_angle = best_outer_angle

        # Multiple restarts for joint optimization
        for restart in range(5):
            if time.time() - start_time > DEADLINE - 5:
                break

            current_params = list(init_params)
            if restart > 0:
                np.random.seed(42 + restart)
                for i in range(len(current_params)):
                    if i < N * 3:
                        current_params[i] += np.random.uniform(-0.03, 0.03)

            try:
                res = minimize(
                    lambda p: objective_joint_smooth(p, temperature=0.01),
                    current_params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )

                inner_list = []
                for i in range(N):
                    inner_list.append([res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]])
                oc = (res.x[N * 3], res.x[N * 3 + 1])
                oa = res.x[N * 3 + 2]

                side = compute_required_side(inner_list, oc, oa)

                if check_all_disjoint(inner_list) and check_inside_outer(inner_list, oc, side, oa):
                    if side < best_joint_side:
                        best_joint_side = side
                        best_joint_inner = inner_list
                        best_joint_center = oc
                        best_joint_angle = oa
            except Exception:
                pass

        if best_joint_side < best_side:
            best_side = best_joint_side
            best_inner = best_joint_inner
            best_outer_center = best_joint_center
            best_outer_angle = best_joint_angle

    # Phase 4: Final polish on outer parameters
    if best_inner is not None and time.time() - start_time < DEADLINE - 3:
        try:
            polish_res = minimize(
                lambda p: compute_required_side(best_inner, (p[0], p[1]), p[2]),
                [best_outer_center[0], best_outer_center[1], best_outer_angle],
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2000}
            )
            if polish_res.fun < best_side:
                best_outer_center = (polish_res.x[0], polish_res.x[1])
                best_outer_angle = polish_res.x[2]
                best_side = polish_res.fun
        except Exception:
            pass

    # Final verification and fallback
    if best_inner is None or not check_all_disjoint(best_inner) or not check_inside_outer(best_inner, best_outer_center, best_side, best_outer_angle):
        # Fallback to known valid configuration
        axial_fallback = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                          (2, 0), (1, 1), (0, 2), (-1, 2)]
        pts_fallback = axial_to_cartesian(axial_fallback)
        best_inner = [[x, y, 0.0] for x, y in pts_fallback]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle
