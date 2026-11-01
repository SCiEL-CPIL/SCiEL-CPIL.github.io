"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
DEADLINE = 142.0  # safety margin before 150s kill

# Axial coordinate candidates for initial layouts
CANDIDATE_AXIALS = [
    # Classic hexagonal cluster (7+4)
    [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2),
    ],
    # Alternative ring
    [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2),
    ],
    # Rectangular-ish
    [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1),
    ],
    # Another variation
    [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2),
    ],
    # Star-like
    [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3),
    ],
    # Dense cluster
    [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
    ],
    # Compact 3x3 + 2
    [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, -1), (2, -1),
    ],
    # Linear-ish with offsets
    [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (1, 2), (2, 2), (3, 2),
        (1, -1),
    ],
]


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


def axial_to_cartesian(axial_list, scale=1.0 + 1e-6):
    """Convert axial coordinates to cartesian with slight scaling to avoid touching."""
    pts = []
    for q, r in axial_list:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        pts.append((x, y))
    return pts


def compute_required_side_smooth(inner_list, outer_center, outer_angle_deg, temperature=0.01):
    """
    Smooth approximation of compute_required_side using log-sum-exp.
    Properly handles both positive and negative projections (absolute value).
    """
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
                all_projections.append(-proj)

    if len(all_projections) == 0:
        return 0.0

    proj_array = np.array(all_projections, dtype=np.float64)
    smooth_max = logsumexp(proj_array, temperature)
    return 2.0 * smooth_max / SQRT3


def objective_joint_smooth(params, temperature=0.01):
    """
    Joint objective for inner positions/rotations and outer parameters.
    params: [x1, y1, a1, ..., xN, yN, aN, cx, cy, cangle]
    """
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

    # Penalty for overlaps (using center distance as proxy for speed in smooth phase)
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


def construct():
    start_time = time.time()

    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # Try multiple candidate initial layouts
    for idx, axial_coords in enumerate(CANDIDATE_AXIALS):
        if time.time() - start_time > DEADLINE - 18:
            break

        centers = axial_to_cartesian(axial_coords)
        initial_inner = [[x, y, 0.0] for x, y in centers]

        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)

        # Phase 1: Optimize outer hexagon for fixed inner layout
        best_val = float('inf')
        best_result = None

        # Try multiple initial angles
        init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]

        for init_angle in init_angles:
            if time.time() - start_time > DEADLINE - 22:
                break

            init_params = [cx_avg, cy_avg, float(init_angle)]

            try:
                res = minimize(
                    lambda p: compute_required_side_smooth(initial_inner, (p[0], p[1]), p[2], temperature=0.01),
                    init_params,
                    method='BFGS',
                    options={'maxiter': 200, 'gtol': 1e-8}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_result = res.x
            except Exception:
                pass

        if best_result is not None:
            cur_outer_center = (best_result[0], best_result[1])
            cur_outer_angle = best_result[2]
            cur_side = compute_required_side(initial_inner, cur_outer_center, cur_outer_angle)
        else:
            cur_outer_center = (cx_avg, cy_avg)
            cur_outer_angle = 0.0
            cur_side = compute_required_side(initial_inner, cur_outer_center, cur_outer_angle)

        # Phase 2: Polish outer parameters with Nelder-Mead
        if time.time() - start_time < DEADLINE - 18:
            try:
                polish_res = minimize(
                    lambda p: compute_required_side(initial_inner, (p[0], p[1]), p[2]),
                    [cur_outer_center[0], cur_outer_center[1], cur_outer_angle],
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2000}
                )
                if polish_res.fun < cur_side:
                    cur_outer_center = (polish_res.x[0], polish_res.x[1])
                    cur_outer_angle = polish_res.x[2]
                    cur_side = polish_res.fun
            except Exception:
                pass

        # Phase 3: Joint optimization of inner positions/rotations and outer parameters
        init_params = []
        for x, y, a in initial_inner:
            init_params.extend([x, y, a])
        init_params.extend([cur_outer_center[0], cur_outer_center[1], cur_outer_angle])

        best_joint_side = cur_side
        best_joint_inner = [list(x) for x in initial_inner]
        best_joint_center = cur_outer_center
        best_joint_angle = cur_outer_angle

        # Multiple restarts for joint optimization
        for restart in range(10):
            if time.time() - start_time > DEADLINE - 18:
                break

            current_params = list(init_params)
            if restart > 0:
                np.random.seed(42 + restart + idx * 100)
                for i in range(len(current_params)):
                    if i < N * 3:
                        current_params[i] += np.random.uniform(-0.10, 0.10)

            try:
                res = minimize(
                    lambda p: objective_joint_smooth(p, temperature=0.01),
                    current_params,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
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

        # Phase 4: Final polish on best joint solution
        if best_joint_side < cur_side and time.time() - start_time < DEADLINE - 8:
            try:
                polish_params = [best_joint_center[0], best_joint_center[1], best_joint_angle]
                polish_res = minimize(
                    lambda p: compute_required_side(best_joint_inner, (p[0], p[1]), p[2]),
                    polish_params,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 8000}
                )
                if polish_res.fun < best_joint_side:
                    best_joint_center = (polish_res.x[0], polish_res.x[1])
                    best_joint_angle = polish_res.x[2]
                    best_joint_side = polish_res.fun
            except Exception:
                pass

            # Additional high-precision polish pass
            if time.time() - start_time < DEADLINE - 5:
                try:
                    polish_params2 = [best_joint_center[0], best_joint_center[1], best_joint_angle]
                    polish_res2 = minimize(
                        lambda p: compute_required_side(best_joint_inner, (p[0], p[1]), p[2]),
                        polish_params2,
                        method='Nelder-Mead',
                        options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 12000}
                    )
                    if polish_res2.fun < best_joint_side:
                        best_joint_center = (polish_res2.x[0], polish_res2.x[1])
                        best_joint_angle = polish_res2.x[2]
                        best_joint_side = polish_res2.fun
                except Exception:
                    pass

        # Update global best
        if best_joint_side < best_side:
            best_side = best_joint_side
            best_inner = best_joint_inner
            best_outer_center = best_joint_center
            best_outer_angle = best_joint_angle

    # Fallback if no valid solution found
    if best_inner is None:
        axial_coords = CANDIDATE_AXIALS[0]
        scale_fallback = 1.05
        fallback_centers = []
        for q, r in axial_coords:
            x = 1.5 * scale_fallback * q
            y = SQRT3 * scale_fallback * (r + q / 2.0)
            fallback_centers.append((x, y))

        best_inner = [[x, y, 0.0] for x, y in fallback_centers]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    # Final verification
    if not check_all_disjoint(best_inner) or not check_inside_outer(best_inner, best_outer_center, best_side, best_outer_angle):
        # Try to fix with a simple scale-up if invalid
        scale_fix = 1.01
        axial_coords = CANDIDATE_AXIALS[0]
        fixed_centers = []
        for q, r in axial_coords:
            x = 1.5 * scale_fix * q
            y = SQRT3 * scale_fix * (r + q / 2.0)
            fixed_centers.append((x, y))

        best_inner = [[x, y, 0.0] for x, y in fixed_centers]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

        # If still invalid, try even larger scale
        if not check_all_disjoint(best_inner) or not check_inside_outer(best_inner, best_outer_center, best_side, best_outer_angle):
            scale_fix2 = 1.02
            fixed_centers2 = []
            for q, r in axial_coords:
                x = 1.5 * scale_fix2 * q
                y = SQRT3 * scale_fix2 * (r + q / 2.0)
                fixed_centers2.append((x, y))

            best_inner = [[x, y, 0.0] for x, y in fixed_centers2]
            best_outer_center = (0.0, 0.0)
            best_outer_angle = 0.0
            best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle
